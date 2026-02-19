"""Pipeline topology DSL example.

Demonstrates using the >> operator to build agent pipelines
with sequential and parallel execution, including Mermaid
visualization.

Usage:
    python examples/concepts/multi_agent/pipeline_topology.py
"""

from sktk import SKTKAgent
from sktk.team.topology import SequentialNode


def main() -> None:
    researcher = SKTKAgent(name="researcher", instructions="Research topics.")
    analyst = SKTKAgent(name="analyst", instructions="Analyze data.")
    editor = SKTKAgent(name="editor", instructions="Edit content.")
    reviewer = SKTKAgent(name="reviewer", instructions="Review quality.")
    synthesizer = SKTKAgent(name="synthesizer", instructions="Synthesize results.")

    pipeline = researcher >> analyst >> [editor, reviewer] >> synthesizer

    print("Pipeline topology (Mermaid):")
    print(pipeline.visualize())

    print(f"\nPipeline type: {type(pipeline).__name__}")
    print(f"Pipeline is a SequentialNode: {isinstance(pipeline, SequentialNode)}")


if __name__ == "__main__":
    main()
