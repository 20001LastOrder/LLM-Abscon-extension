# parser_caller.py
import json
import shutil
import pythonmonkey as pm

import networkx as nx
import asyncio
from pathlib import Path

folder = Path(__file__).parent.parent
parse_mermaid_js = pm.require(f"{folder}/js/native_parser.bundle.js")

NODE = shutil.which("node") or "node"  # help on Windows PATH issues


class MermaidGraphParser:
    def parse(self, mermaid_text: str) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph_json = asyncio.run(parse_mermaid_py(mermaid_text))

        for node_id, (raw_label, node) in enumerate(graph_json["vertices"].items()):
            graph.add_node(raw_label, label=node.get("text", raw_label), id=node_id)
        for edge in graph_json["edges"]:
            start = edge["start"]
            end = edge["end"]

            if edge.get("text", ""):
                graph.add_edge(start, end, label=edge["text"])
            else:
                graph.add_edge(start, end)

        return graph


async def parse_mermaid_py(src: str):
    # Use top-level await inside this eval call
    s = await parse_mermaid_js["parse_mermaid"](src)
    return json.loads(s)


if __name__ == "__main__":
    mermaid_graph = """
    graph TD
        StartNode --> DetectIncident["Detect incident"]
        DetectIncident --> CheckFinancialImpact{"Check financial impact?"}
        CheckFinancialImpact -->|No| EndNode["End process"]
        CheckFinancialImpact -->|Yes| OpenEMS["Open EMS"]
        OpenEMS --> ContactApp["Contact Application Team"]
        ContactApp --> ContactSA["Contact SA Team"]
        ContactSA --> ContactDBA["Contact DBA Team"]
        ContactDBA --> OpenConference["Open conference line"]
        OpenConference --> Troubleshoot["Troubleshoot the issue"]
        Troubleshoot --> CheckResolution{"Check if resolution is known?"}
        
        CheckResolution -->|Yes| FixIssue["Fix the issue"]
        FixIssue --> ResolveEMS["Resolve and close EMS"] --> EndNode
        
        CheckResolution -->|No| CheckVendor{"Check vendor for fix?"}
        CheckVendor -->|Yes| FixVendor["Fix the issue (vendor fix)"] --> ResolveEMS
        CheckVendor -->|No| Failover["Failover to COB"] --> ResolveEMS
    """
    import asyncio

    result = asyncio.run(parse_mermaid_py(mermaid_graph))
    print(result.keys())
    print(result["vertices"])
