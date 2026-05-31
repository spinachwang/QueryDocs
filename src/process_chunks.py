import os
import re
from pathlib import Path
from src.text_splitter import TextSplitter


def process_all_reports(
    input_dir: Path,
    output_dir: Path,
    chunk_size: int = 300,
    overlap: int = 50
):
    """批量处理所有年报的分块

    Args:
        input_dir: 包含各年报目录的目录（如 03_reports_markdown）
        output_dir: 分块输出目录
        chunk_size: 每个chunk的目标token数
        overlap: 相邻chunk之间的overlap token数
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splitter = TextSplitter()

    # 遍历所有子目录
    processed = 0
    skipped = 0

    for subdir in input_dir.iterdir():
        if not subdir.is_dir():
            continue

        # 跳过拆分的目录（如 _p1-200, _p201-222）
        if re.search(r'_p\d+-\d+$', subdir.name):
            print(f"跳过拆分目录: {subdir.name}")
            skipped += 1
            continue

        # 查找 *_content_list_v2.json
        content_v2_files = list(subdir.glob("*_content_list_v2.json"))
        if not content_v2_files:
            print(f"未找到 content_list_v2.json: {subdir.name}")
            continue

        content_v2_path = content_v2_files[0]
        output_path = output_dir / f"{subdir.name}_chunks.json"

        print(f"处理: {subdir.name} -> {output_path.name}")

        try:
            splitter.split_content_list_v2(
                content_v2_path,
                output_path,
                chunk_size=chunk_size,
                overlap=overlap,
                pdf_name=subdir.name
            )
            processed += 1
        except Exception as e:
            print(f"处理失败 {subdir.name}: {e}")

    print(f"\n完成! 成功: {processed}, 跳过: {skipped}")


if __name__ == "__main__":
    from pathlib import Path

    input_dir = Path("data/stock_data/debug_data/03_reports_markdown")
    output_dir = Path("data/stock_data/debug_data/chunked_reports")

    process_all_reports(input_dir, output_dir, chunk_size=500, overlap=100)