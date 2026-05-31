import json
import tiktoken
from pathlib import Path
from typing import List, Dict
import os
import re

# 原子类型：不可切断
ATOMIC_TYPES = {'title', 'table', 'image', 'page_header', 'page_number'}

# 可分割类型：可以按自然边界切断
SPLITTABLE_TYPES = {'paragraph', 'list'}


class TextSplitter():
    def _extract_text_from_item(self, item: Dict) -> str:
        """从item中提取文本内容"""
        item_type = item.get('type')
        content = item.get('content', {})

        if item_type == 'title':
            texts = []
            for t in content.get('title_content', []):
                if t.get('type') == 'text':
                    texts.append(t.get('content', ''))
            return '\n'.join(texts)

        elif item_type == 'paragraph':
            texts = []
            for t in content.get('paragraph_content', []):
                if t.get('type') == 'text':
                    texts.append(t.get('content', ''))
            return '\n'.join(texts)

        elif item_type == 'list':
            items = content.get('list_items', [])
            texts = []
            for item in items:
                if item.get('item_type') == 'text':
                    for t in item.get('item_content', []):
                        if t.get('type') == 'text':
                            texts.append(t.get('content', ''))
                elif item.get('item_type') == 'table':
                    table_text = self._extract_table_text(item.get('item_content', {}))
                    texts.append(table_text)
            return '\n'.join(texts)

        elif item_type == 'table':
            return self._extract_table_text(content)

        elif item_type == 'image':
            img_path = content.get('image_source', {}).get('path', '')
            return f'[IMAGE: {img_path}]'

        return ''

    def _extract_table_text(self, content: Dict) -> str:
        """从table的content中提取文本内容"""
        parts = []

        caption = content.get('table_caption', [])
        if caption:
            caption_texts = [c.get('content', '') for c in caption if c.get('type') == 'text']
            if caption_texts:
                parts.append(' '.join(caption_texts))

        html = content.get('html', '')
        if html:
            parts.append(html)

        footnote = content.get('table_footnote', [])
        if footnote:
            footnote_texts = [f.get('content', '') for f in footnote if f.get('type') == 'text']
            if footnote_texts:
                parts.append(' '.join(footnote_texts))

        return '\n'.join(parts) if parts else '[TABLE]'

    def _split_long_text(self, text: str, max_tokens: int) -> List[str]:
        """将长文本按句子分割成多个片段，每个不超过max_tokens"""
        enc = tiktoken.get_encoding('o200k_base')
        sentences = re.split(r'([。！？；\n])', text)
        result = []
        buffer = []
        buffer_tokens = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            sent_tokens = len(enc.encode(sent))
            next_tokens = buffer_tokens + sent_tokens

            if next_tokens > max_tokens and buffer:
                result.append(''.join(buffer))
                buffer = [sent]
                buffer_tokens = sent_tokens
            else:
                buffer.append(sent)
                buffer_tokens = next_tokens

        if buffer:
            result.append(''.join(buffer))

        return result

    def split_content_list_v2(self, content_list_v2_path: Path, output_path: Path,
                                chunk_size: int = 300, overlap: int = 50,
                                pdf_name: str = None) -> None:
        """处理MinerU的content_list_v2.json，按标题分块

        标题不单独成chunk，而是与下一个paragraph/table/list/image组合。
        如果内容过长需要分割，标题会作为前缀保留在第一个子chunk中。

        Args:
            content_list_v2_path: content_list_v2.json文件路径
            output_path: 输出json文件路径
            chunk_size: 每个chunk的目标token数
            overlap: 相邻chunk之间的overlap token数
            pdf_name: PDF文件名（用于metainfo）
        """
        with open(content_list_v2_path, 'r', encoding='utf-8') as f:
            pages = json.load(f)

        if pdf_name is None:
            pdf_name = content_list_v2_path.parent.name

        enc = tiktoken.get_encoding('o200k_base')

        all_chunks = []
        chunk_id = 0

        # 当前累积的chunk内容（带标题前缀）
        current_title = ""
        current_parts = []  # [(text, is_new_title_part), ...]
        current_tokens = 0
        current_page = 0

        def flush():
            """输出当前chunk"""
            nonlocal chunk_id, current_parts, current_tokens
            if not current_parts:
                return

            # 组装文本
            chunk_text = ''.join(current_parts).strip()
            tokens = len(enc.encode(chunk_text))

            all_chunks.append({
                'id': chunk_id,
                'page': current_page,
                'type': 'content',
                'text': chunk_text,
                'length_tokens': tokens
            })
            chunk_id += 1
            current_parts = []
            current_tokens = 0

        def add_content(text: str, page: int):
            """添加内容到当前chunk"""
            nonlocal current_parts, current_tokens, current_page
            current_parts.append(text)
            current_tokens += len(enc.encode(text))
            current_page = page

        def split_and_add(text: str, page: int):
            """分割长文本，按句子切割，必要时创建新chunk"""
            nonlocal current_title, current_parts, current_tokens
            if current_tokens > 0:
                flush()

            parts = self._split_long_text(text, chunk_size)
            for i, part in enumerate(parts):
                if i == 0 and current_title:
                    # 第一个子chunk带标题前缀
                    add_content(current_title + '\n' + part, page)
                    current_title = ""
                else:
                    add_content(part, page)
                if current_tokens >= chunk_size:
                    flush()

        # 遍历所有页面
        for page_num, page in enumerate(pages):
            page_num = page_num + 1  # 1-indexed

            for item in page:
                item_type = item.get('type')
                text = self._extract_text_from_item(item)

                if not text or item_type in ('page_header', 'page_number'):
                    continue

                # 标题：存储但不立即输出
                if item_type == 'title':
                    # 如果有待组合的标题且内容超过阈值，先flush
                    if current_title and current_tokens > 30:
                        flush()
                    current_title = text
                    continue

                # 内容类型
                if item_type in SPLITTABLE_TYPES or item_type in ATOMIC_TYPES:
                    item_tokens = len(enc.encode(text))

                    if current_title:
                        # 有待组合的标题
                        combined_text = current_title + '\n' + text
                        combined_tokens = len(enc.encode(combined_text))
                        current_title = ""

                        if combined_tokens > chunk_size:
                            # 先flush当前
                            if current_parts:
                                flush()
                            # 分割内容
                            split_and_add(text, page_num)
                        else:
                            if current_tokens + combined_tokens > chunk_size and current_parts:
                                flush()
                            add_content(combined_text, page_num)

                            # 检查是否超size
                            while current_tokens > chunk_size and len(current_parts) > 1:
                                # 把最后一个part分割出去
                                last_part = current_parts.pop()
                                current_tokens -= len(enc.encode(last_part))

                                # 处理最后一个part
                                if current_tokens > 0:
                                    flush()

                                split_and_add(
                                    last_part.replace(current_title + '\n', '', 1) if current_title else last_part,
                                    page_num
                                )

                            # 如果仍然超size且只有一个part，说明这个item本身太长
                            # 需要强制分割（按HTML标签或直接截断）
                            if current_tokens > chunk_size and len(current_parts) == 1:
                                part = current_parts[0]
                                # 直接按HTML标签分割表格
                                html_parts = re.split(r'(<tr>|</tr>|<td>|</td>)', part)
                                sub_buffer = []
                                sub_tokens = 0
                                first = True
                                for hp in html_parts:
                                    hp_tokens = len(enc.encode(hp))
                                    if sub_tokens + hp_tokens > chunk_size and sub_buffer:
                                        # 输出当前子chunk
                                        sub_text = ''.join(sub_buffer)
                                        if first and current_title:
                                            sub_text = current_title + '\n' + sub_text
                                            current_title = ""
                                        flush()
                                        all_chunks.append({
                                            'id': chunk_id,
                                            'page': page_num,
                                            'type': 'content',
                                            'text': sub_text,
                                            'length_tokens': sub_tokens
                                        })
                                        chunk_id += 1
                                        sub_buffer = [hp]
                                        sub_tokens = hp_tokens
                                        first = False
                                    else:
                                        sub_buffer.append(hp)
                                        sub_tokens += hp_tokens
                                # 处理剩余
                                if sub_buffer:
                                    sub_text = ''.join(sub_buffer)
                                    if first and current_title:
                                        sub_text = current_title + '\n' + sub_text
                                        current_title = ""
                                    flush()
                                    all_chunks.append({
                                        'id': chunk_id,
                                        'page': page_num,
                                        'type': 'content',
                                        'text': sub_text,
                                        'length_tokens': sub_tokens
                                    })
                                    chunk_id += 1
                                current_parts = []
                                current_tokens = 0
                    else:
                        # 没有pending标题
                        if current_tokens + item_tokens > chunk_size and current_parts:
                            flush()

                        add_content(text, page_num)

                        # 检查是否超size
                        while current_tokens > chunk_size and len(current_parts) > 1:
                            last_part = current_parts.pop()
                            current_tokens -= len(enc.encode(last_part))

                            if current_tokens > 0:
                                flush()

                            split_and_add(last_part, page_num)

        # 处理最后一个chunk
        if current_parts:
            flush()

        # 输出结果
        result = {
            'metainfo': {
                'source': pdf_name,
                'total_chunks': len(all_chunks)
            },
            'content': {
                'chunks': all_chunks
            }
        }

        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"分块完成: {len(all_chunks)} chunks -> {output_path}")