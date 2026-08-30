import os
import sys
import time
import asyncio
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("test_simulator")

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class SimpleConsole:
        def print(self, *args, **kwargs):
            clean = " ".join(str(a) for a in args)
            import re
            clean = re.sub(r'\[.*?\]', '', clean)
            print(clean)
        def clear(self):
            pass
    console = SimpleConsole()

from agent import VoiceAIAgentSession

async def run_simulation():
    if HAS_RICH:
        console.clear()
        console.print(Panel.fit(
            "[bold cyan]AI Voice Agent - Local Conversational Simulator[/bold cyan]\n"
            "[dim]Full Glass-to-Glass Latency Benchmark & Database Handshake[/dim]",
            border_style="cyan"
        ))
    else:
        print("\n========================================================")
        print("    AI Voice Agent - Local Conversational Simulator     ")
        print("    Full Glass-to-Glass Latency & Database Handshake    ")
        print("========================================================\n")

    # 1. Initialize Active Voice Session
    session = VoiceAIAgentSession(
        room_name="local-sim-room-01",
        caller_did="+14156390491",
        customer_phone="+15558902341",
    )
    
    console.print("[yellow]► Step 1: Handshaking with Go Backend (:8080)...[/yellow]")
    await session.fetch_backend_context()
    
    if HAS_RICH:
        table = Table(title="Live Session Context", show_header=True, header_style="bold green")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Tenant ID", str(session.tenant_id))
        table.add_row("Assigned Agent", session.agent_name)
        table.add_row("Kokoro Voice Persona", f"{session.voice_name} ({session.voice_speed}x speed)")
        table.add_row("Inbound DID", session.caller_did)
        table.add_row("Customer Phone", session.customer_phone)
        console.print(table)
    else:
        print(f"  • Tenant ID: {session.tenant_id}")
        print(f"  • Assigned Agent: {session.agent_name}")
        print(f"  • Kokoro Voice Persona: {session.voice_name} ({session.voice_speed}x speed)")
        print(f"  • Inbound DID: {session.caller_did}")
        print(f"  • Customer Phone: {session.customer_phone}")

    console.print("\n[bold green]► Step 2: Conversation Active! Type your message to the agent (or 'exit' to end call):[/bold green]")
    
    # Pre-built quick test scenarios
    console.print("[dim]Quick Test Suggestions:[/dim]")
    console.print("  [dim]1. 'Hi, what is your commercial solar warranty?' (Tests pgvector RAG)[/dim]")
    console.print("  [dim]2. 'Can you text me your pricing brochure?' (Tests Live SMS Push)[/dim]")
    console.print("  [dim]3. 'I want to schedule an appointment for Tuesday.' (Tests Calendar Booking)[/dim]\n")

    loop = asyncio.get_event_loop()
    
    while True:
        try:
            user_input = await loop.run_in_executor(None, input, "\n[You]: ")
            user_input = user_input.strip()
            
            if not user_input or user_input.lower() in ["exit", "quit", "bye", "hangup"]:
                console.print("[yellow]Hanging up call...[/yellow]")
                break
                
            t0 = time.perf_counter()
            reply = await session.process_user_turn(user_input)
            total_turn_ms = (time.perf_counter() - t0) * 1000
            
            console.print(f"[bold magenta][{session.agent_name}]:[/bold magenta] {reply}")
            console.print(f"[dim green]└── Turn Latency: {total_turn_ms:.1f}ms | Audio Generated with {session.voice_name}[/dim green]")
            
        except (KeyboardInterrupt, EOFError):
            break

    # 3. Finalize Call & Atomic Billing
    console.print("\n[yellow]► Step 3: Finalizing Call in Go Backend & Updating Credits in PostgreSQL...[/yellow]")
    await session.end_session()
    console.print("[bold green]✓ Call Completed Successfully! Check Dashboard at http://localhost:3000/voice-recorder and http://localhost:3000/billing[/bold green]\n")

if __name__ == "__main__":
    asyncio.run(run_simulation())
