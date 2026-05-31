import click
from pathlib import Path
from src.pipeline import Pipeline, configs, preprocess_configs
from src.pdf_mineru import MinerUParser
import os
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)


@click.group()
def cli():
    """Pipeline command line interface for processing PDF reports and questions."""
    pass


@cli.command()
@click.option('--parallel/--sequential', default=True, help='Run parsing in parallel or sequential mode')
@click.option('--chunk-size', default=2, help='Number of PDFs to process in each worker')
@click.option('--max-workers', default=10, help='Number of parallel worker processes')
def parse_pdfs(parallel, chunk_size, max_workers):
    """Parse PDF reports with optional parallel processing using MinerU API."""
    root_path = Path.cwd()

    # 从环境变量获取 API Key
    api_key = os.environ.get("MINERU_API_KEY")
    if not api_key:
        click.echo("错误: 请设置环境变量 MINERU_API_KEY")
        return

    click.echo(f"使用 MinerU API 解析 PDFs (parallel={parallel})")

    # 输出目录
    output_dir = root_path / "data" / "stock_data" / "debug_data" / "03_reports_markdown"

    # 获取所有PDF
    pdf_dir = root_path / "data" / "stock_data" / "pdf_reports"
    pdf_paths = list(pdf_dir.glob("*.pdf"))
    click.echo(f"找到 {len(pdf_paths)} 个PDF文件")

    # 批量解析
    parser = MinerUParser(api_key=api_key, output_dir=output_dir)
    paths = parser.parse(pdf_paths)
    click.echo(f"解析完成，输出到 {output_dir}")


@cli.command()
@click.option('--max-workers', default=10, help='Number of workers for table serialization')
def serialize_tables(max_workers):
    """Serialize tables in parsed reports using parallel threading."""
    root_path = Path.cwd()
    pipeline = Pipeline(root_path)

    click.echo(f"Serializing tables (max_workers={max_workers})...")
    pipeline.serialize_tables(max_workers=max_workers)


@cli.command()
@click.option('--config', type=click.Choice(['ser_tab', 'no_ser_tab']), default='no_ser_tab', help='Configuration preset to use')
def process_reports(config):
    """Process parsed reports through the pipeline stages."""
    root_path = Path.cwd()
    run_config = preprocess_configs[config]
    pipeline = Pipeline(root_path, run_config=run_config)

    click.echo(f"Processing parsed reports (config={config})...")
    pipeline.process_parsed_reports()


@cli.command()
def build_vectors():
    """Build vector databases from chunked reports."""
    from src.ingestion import VectorDBIngestor

    chunk_dir = Path("data/stock_data/debug_data/chunked_reports")
    output_dir = Path("data/stock_data/debug_data/databases/vector_dbs")
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Building vectors from {chunk_dir}...")
    ingestor = VectorDBIngestor()
    ingestor.process_reports(chunk_dir, output_dir)
    click.echo(f"Vectors built in {output_dir}")


@cli.command()
def build_bm25():
    """Build BM25 indexes from chunked reports."""
    from src.ingestion import BM25Ingestor

    chunk_dir = Path("data/stock_data/debug_data/chunked_reports")
    output_dir = Path("data/stock_data/debug_data/databases/bm25_dbs")
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Building BM25 indexes from {chunk_dir}...")
    ingestor = BM25Ingestor()
    ingestor.process_reports(chunk_dir, output_dir)
    click.echo(f"BM25 indexes built in {output_dir}")


@cli.command()
@click.option('--config', type=click.Choice(['base', 'pdr', 'max', 'max_no_ser_tab', 'max_nst_o3m', 'max_st_o3m', 'ibm_llama70b', 'ibm_llama8b', 'gemini_thinking', 'minimax']), default='base', help='Configuration preset to use')
def process_questions(config):
    """Process questions using the pipeline."""
    root_path = Path.cwd()
    run_config = configs[config]
    pipeline = Pipeline(root_path, run_config=run_config)

    click.echo(f"Processing questions (config={config})...")
    pipeline.process_questions()


if __name__ == '__main__':
    cli()