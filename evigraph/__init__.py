"""EviGraph: evidence-graph frame selection for long-video question answering.

Phase 1 (graph_builder) builds a question-conditioned evidence graph with a VLM;
Phase 2 (selection) turns the graph plus a BLIP-ITM relevance ranking into the
N frames that a frozen answering backbone sees.
"""
__version__ = "1.0.0"
