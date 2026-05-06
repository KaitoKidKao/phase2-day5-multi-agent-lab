"""Command-line entrypoint for the lab starter."""

import os
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init():
    settings = get_settings()
    configure_logging(settings.log_level)
    
    # Export environment variables for LangChain/LangGraph tracing
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        console.print("[dim]LangSmith tracing enabled.[/dim]")

    if settings.langfuse_public_key and settings.langfuse_secret_key:
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
        os.environ["LANGFUSE_HOST"] = settings.langfuse_base_url
        console.print("[dim]Langfuse tracing enabled.[/dim]")


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a real single-agent baseline."""

    _init()
    console.print(f"[bold yellow]Running Single-Agent Baseline for:[/bold yellow] {query}")
    
    llm = LLMClient()
    if not llm.validate_query(query):
        console.print("[bold red]Error:[/bold red] Query violated safety or length limits.")
        raise typer.Exit(code=1)

    response = llm.complete(
        "You are a helpful research assistant. Answer the user query directly in a structured format.",
        query
    )
    
    console.print(Panel.fit(response.content, title="Single-Agent Result"))
    console.print(f"[dim]Cost: ${response.cost_usd:.4f} | Tokens: {response.output_tokens}[/dim]")


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    settings = get_settings()
    console.print(f"[bold blue]Starting Multi-Agent Research for:[/bold blue] {query}")
    
    state = ResearchState(request=ResearchQuery(query=query))
    
    # Initialize services
    llm = LLMClient()
    search = SearchClient()

    workflow = MultiAgentWorkflow(llm, search)
    
    # Setup callbacks
    callbacks = []
    langfuse_handler = None
    try:
        from langfuse.callback import CallbackHandler
        if settings.langfuse_public_key and settings.langfuse_secret_key:
            langfuse_handler = CallbackHandler(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_base_url
            )
            callbacks.append(langfuse_handler)
    except ImportError:
        pass

    try:
        # Pass callbacks to run
        result = workflow.run(state, callbacks=callbacks)
        
        if result.errors:
            for err in result.errors:
                console.print(f"[bold red]Error:[/bold red] {err}")
            
            # Explicitly mark trace as error in Langfuse if possible
            if langfuse_handler:
                try:
                    trace_id = langfuse_handler.get_trace_id()
                    if trace_id:
                        langfuse_handler.langfuse.trace(id=trace_id, status_message="Blocked by Guardrail", release="v1.0")
                except Exception:
                    pass

            if "blocked" in result.route_history:
                raise typer.Exit(code=1)
        
        console.print("\n[bold green]Final Answer:[/bold green]")
        console.print(Panel(result.final_answer or "No answer generated.", title="Multi-Agent Result"))
        
        if result.sources:
            console.print("\n[bold cyan]Sources:[/bold cyan]")
            for src in result.sources:
                console.print(f"- {src.title} ({src.url})")
                
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[bold red]Error during execution:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        # ALWAYS flush traces before exiting
        if langfuse_handler:
            with console.status("[dim]Flushing traces to Langfuse...[/dim]"):
                langfuse_handler.flush()


if __name__ == "__main__":
    app()
