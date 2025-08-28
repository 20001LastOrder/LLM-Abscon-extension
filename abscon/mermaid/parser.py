# parser_caller.py
import json
import shutil
import subprocess

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


NODE = shutil.which("node") or "node"  # help on Windows PATH issues


def parse_mermaid_py(text: str) -> dict:
    try:
        proc = subprocess.run(
            # Note that the path is relative to the project root
            [NODE, "abscon/mermaid/native_parser.mjs"],
            input=json.dumps({"text": text}),
            text=True,
            capture_output=True,
            check=True,
        )
        print(proc.stdout)  # Debug: print the raw output
        return json.loads(proc.stdout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Node error:\n{e.stderr}") from e
