import os
import time
import zipfile
import logging
from pathlib import Path
from typing import List, Dict, Optional
from PyPDF2 import PdfReader, PdfWriter
import requests

_log = logging.getLogger(__name__)

MAX_PAGES = 200


def split_pdf(input_path: str, output_dir: str, max_pages: int = MAX_PAGES) -> List[str]:
    """将大PDF拆分成多个小PDF

    Args:
        input_path: 输入PDF路径
        output_dir: 输出目录
        max_pages: 每部分最大页数

    Returns:
        拆分后的PDF路径列表
    """
    reader = PdfReader(input_path)
    total_pages = len(reader.pages)
    base_name = Path(input_path).stem

    output_paths = []
    for i in range(0, total_pages, max_pages):
        writer = PdfWriter()
        end = min(i + max_pages, total_pages)

        for page_num in range(i, end):
            writer.add_page(reader.pages[page_num])

        output_path = os.path.join(output_dir, f"{base_name}_p{i+1}-{end}.pdf")
        with open(output_path, 'wb') as f:
            writer.write(f)
        output_paths.append(output_path)
        _log.info(f"拆分: {output_path} (页 {i+1}-{end})")

    return output_paths


def get_pdf_page_count(file_path: str) -> int:
    """获取PDF页数"""
    reader = PdfReader(file_path)
    return len(reader.pages)


def check_and_split_pdf(file_path: str, output_dir: str = None) -> List[str]:
    """检查PDF页数，必要时拆分为多个PDF

    Returns:
        PDF路径列表（单个或多个）
    """
    page_count = get_pdf_page_count(file_path)
    _log.info(f"PDF {Path(file_path).name}: {page_count} 页")

    if output_dir is None:
        output_dir = os.path.dirname(file_path)

    if page_count > MAX_PAGES:
        _log.info(f"PDF超过{MAX_PAGES}页，拆分成多个文件")
        return split_pdf(file_path, output_dir, MAX_PAGES)
    else:
        return [file_path]


class MinerUAPI:
    """MinerU API 批量处理PDF"""

    def __init__(self, api_key: str, model_version: str = "vlm"):
        self.api_key = api_key
        self.model_version = model_version
        self.base_url = "https://mineru.net/api/v4"

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def batch_get_upload_urls(self, files: List[Dict[str, str]]) -> tuple:
        """获取批量上传URL

        Args:
            files: [{"name": "file.pdf", "data_id": "unique_id"}, ...]

        Returns:
            (batch_id, file_urls)
        """
        url = f"{self.base_url}/file-urls/batch"
        data = {
            "files": files,
            "model_version": self.model_version
        }
        res = requests.post(url, headers=self._headers(), json=data)
        res.raise_for_status()
        result = res.json()

        if result.get("code") != 0:
            raise RuntimeError(f"获取上传URL失败: {result.get('msg')}")

        data = result["data"]
        return data["batch_id"], data["file_urls"]

    def upload_file(self, file_path: str, upload_url: str) -> bool:
        """上传文件到预签名URL"""
        with open(file_path, 'rb') as f:
            res = requests.put(upload_url, data=f)
        return res.status_code == 200

    def get_batch_results(self, batch_id: str) -> dict:
        """获取批量任务结果"""
        url = f"{self.base_url}/extract-results/batch/{batch_id}"
        res = requests.get(url, headers=self._headers())
        res.raise_for_status()
        return res.json()["data"]

    def wait_for_completion(
        self,
        batch_id: str,
        poll_interval: int = 5,
        timeout: int = 3600
    ) -> dict:
        """轮询等待任务完成"""
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"任务超时 ({timeout}s)")

            result = self.get_batch_results(batch_id)
            _log.info(f"API返回: {result}")

            # 等待有extract_result字段
            if "extract_result" not in result:
                _log.info("等待文件上传中...")
                time.sleep(poll_interval)
                continue

            states = [r["state"] for r in result["extract_result"]]

            _log.info(f"批次状态: {states}")

            # 统计各状态
            done_count = sum(1 for s in states if s == "done")
            failed_count = sum(1 for s in states if s == "failed")
            total = len(states)

            # 所有任务都完成（包括失败的）
            if done_count + failed_count == total:
                return result

            # 还有任务在处理中，等待一下
            time.sleep(poll_interval)

    def download_and_extract(
        self,
        result: dict,
        output_dir: str
    ) -> Dict[str, str]:
        """下载并解压结果

        Returns:
            {data_id: output_path} 映射
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        for r in result["extract_result"]:
            data_id = r["data_id"]
            state = r["state"]

            if state != "done":
                _log.warning(f"跳过 {data_id}: 状态为 {state}")
                continue

            full_zip_url = r.get("full_zip_url")
            if not full_zip_url:
                _log.warning(f"跳过 {data_id}: 无下载链接")
                continue

            # 下载ZIP
            local_zip = os.path.join(output_dir, f"{data_id}.zip")
            _log.info(f"下载 {data_id}: {full_zip_url}")

            with requests.get(full_zip_url, stream=True) as resp:
                resp.raise_for_status()
                with open(local_zip, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)

            # 解压
            extract_dir = os.path.join(output_dir, data_id)
            with zipfile.ZipFile(local_zip, 'r') as z:
                z.extractall(extract_dir)

            # 清理ZIP
            os.remove(local_zip)

            paths[data_id] = extract_dir
            _log.info(f"完成 {data_id}: {extract_dir}")

        return paths


class MinerUParser:
    """MinerU 批量解析PDF"""

    def __init__(
        self,
        api_key: str,
        output_dir: Path = Path("./parsed_pdfs"),
        model_version: str = "vlm"
    ):
        self.api = MinerUAPI(api_key, model_version)
        self.output_dir = Path(output_dir)

    def parse(self, pdf_paths: List[Path]) -> Dict[str, str]:
        """批量解析PDF文件

        Args:
            pdf_paths: PDF文件路径列表

        Returns:
            {data_id: output_dir} 映射
        """
        if not pdf_paths:
            return {}

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 检查并拆分超过200页的PDF
        all_pdf_paths = []
        for path in pdf_paths:
            split_paths = check_and_split_pdf(str(path), str(self.output_dir))
            all_pdf_paths.extend(split_paths)

        # 准备文件列表
        files = [
            {"name": Path(p).name, "data_id": Path(p).stem}
            for p in all_pdf_paths
        ]

        _log.info(f"待处理 {len(files)} 个PDF文件")

        # 获取上传URL
        batch_id, file_urls = self.api.batch_get_upload_urls(files)

        # 上传文件
        for path, url in zip(all_pdf_paths, file_urls):
            if not self.api.upload_file(str(path), url):
                raise RuntimeError(f"上传失败: {path}")

        # 等待完成
        result = self.api.wait_for_completion(batch_id)

        # 下载解压
        extracted_paths = self.api.download_and_extract(result, str(self.output_dir))

        # 合并拆分后的PDF
        if all_pdf_paths:
            split_dirs = [str(self.output_dir / Path(p).stem) for p in all_pdf_paths if '_p' in str(p)]
            if len(split_dirs) > 1:
                self.merge_split_pdfs(split_dirs)

        return extracted_paths

    def merge_split_pdfs(self, split_dirs: List[str]) -> Dict[str, str]:
        """合并拆分后的PDF解析结果

        Args:
            split_dirs: 拆分目录列表，如 ['【财报】中芯国际2024年报_p1-200', '【财报】中芯国际2024年报_p201-222']

        Returns:
            {merged_name: merged_path} 映射
        """
        import json
        import re
        from collections import defaultdict

        # 按原始名分组
        groups = defaultdict(list)
        pattern = re.compile(r'^(.*)_p\d+-\d+$')

        for d in split_dirs:
            base_name = Path(d).stem
            match = pattern.match(base_name)
            if match:
                original_name = match.group(1)
            else:
                original_name = base_name
            groups[original_name].append(d)

        merged_paths = {}

        for original_name, dirs in groups.items():
            if len(dirs) == 1:
                # 没有拆分，直接跳过
                _log.info(f"无需合并: {original_name}")
                continue

            _log.info(f"合并: {original_name} ({len(dirs)} 个分片)")

            # 创建输出目录
            merged_dir = os.path.join(self.output_dir, original_name)
            os.makedirs(merged_dir, exist_ok=True)

            # 排序分片
            def get_page_start(d):
                match = re.search(r'_p(\d+)-\d+$', Path(d).stem)
                return int(match.group(1)) if match else 0

            dirs.sort(key=get_page_start)

            # 合并full.md
            full_md_contents = []
            for d in dirs:
                md_path = os.path.join(d, 'full.md')
                if os.path.exists(md_path):
                    with open(md_path, 'r', encoding='utf-8') as f:
                        full_md_contents.append(f.read())

            if full_md_contents:
                merged_md_path = os.path.join(merged_dir, 'full.md')
                with open(merged_md_path, 'w', encoding='utf-8') as f:
                    f.write('\n\n---\n\n'.join(full_md_contents))
                _log.info(f"合并full.md: {merged_md_path}")

            # 合并images
            images_dir = os.path.join(merged_dir, 'images')
            os.makedirs(images_dir, exist_ok=True)

            for d in dirs:
                page_range = re.search(r'_p(\d+-\d+)$', Path(d).stem).group(1)
                src_images = os.path.join(d, 'images')
                if os.path.isdir(src_images):
                    for img in os.listdir(src_images):
                        # 加上页码范围前缀避免冲突
                        new_name = f"{page_range}_{img}"
                        src_path = os.path.join(src_images, img)
                        dst_path = os.path.join(images_dir, new_name)
                        import shutil
                        shutil.copy2(src_path, dst_path)
            _log.info(f"合并images: {images_dir}")

            # 合并JSON文件
            json_files = ['layout.json', '_content_list.json', '_content_list_v2.json', '_model.json']

            for json_file in json_files:
                json_contents = []
                for d in dirs:
                    # 匹配带uuid前缀的json文件
                    pattern_json = f"*{json_file}"
                    import glob
                    matches = glob.glob(os.path.join(d, pattern_json))
                    for match_path in matches:
                        with open(match_path, 'r', encoding='utf-8') as f:
                            try:
                                json_contents.append(json.load(f))
                            except:
                                pass

                if json_contents and isinstance(json_contents[0], list):
                    # 合并JSON数组
                    merged = []
                    for c in json_contents:
                        merged.extend(c)

                    # 写入合并后的文件
                    base_name = json_file.lstrip('_')
                    out_path = os.path.join(merged_dir, f"{original_name}_{base_name}")
                    with open(out_path, 'w', encoding='utf-8') as f:
                        json.dump(merged, f, ensure_ascii=False, indent=2)
                    _log.info(f"合并{json_file}: {out_path}")

            merged_paths[original_name] = merged_dir
            _log.info(f"合并完成: {original_name} -> {merged_dir}")

        return merged_paths


def parse_pdf_reports(
    pdf_dir: Path,
    output_dir: Path,
    api_key: str,
    model_version: str = "vlm"
) -> Dict[str, str]:
    """便捷函数：解析目录下所有PDF"""
    pdf_paths = list(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        _log.warning(f"目录中无PDF文件: {pdf_dir}")
        return {}

    parser = MinerUParser(api_key, output_dir, model_version)
    return parser.parse(pdf_paths)