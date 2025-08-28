from langchain_core.prompts.chat import ChatPromptTemplate

PROMPT_SIMPLE = """You are an expert software modeler who can create an UML activity diagram from a text description. Given a natural language description, construct an activity diagram by identifying the activities and decisions. Then connect them together to represent the activity flow. The activity diagram are represented as mermaid graphs. The activity diagram contains the following three types of node:
1. Dctivity node. e.g. activity A happens and then activity B happens
```mermaid
graph LR
1["A"] --> 2["B"] 
```

2. Decision node. e.g. activity A happens if Decision D is Yes otherwise activity B happens
```mermaid
graph LR
1{{"D"}} --> |"Yes"| 2["A"]
1 --> |"No"| 3["B"]
```

3. Fork and Join node. e.g. activity A and B happens at the same time after activity F. After A and B finishes, activity J happens
```mermaid
graph LR
1["F"] --> 2["A"]
1 --> 3["B"]
2 --> 4["J"]
3 --> 4
```

Notice that the activity diagram should respect the following constraints:
1. All outgoing edges of a decision node must have a condition
2. There can only be one initial node
3. The activity diagram should be connected
4. The initial node should be able to reach all nodes in the activity diagram
5. Use numbers (1..N) to represent node IDs as shown in the example


First identifying all activities and decisions, then describe how they can be connected, finally output the activity diagram as a mermaid graph:
```mermaid
<result>
```

Description:
{user_input}
"""

PROMPT_SIMPLE_WITH_EXAMPLES = """You are an expert software modeler who can create an UML activity diagram from a text description. Given a natural language description, construct an activity diagram by identifying the activities and decisions. Then connect them together to represent the activity flow. The activity diagram are represented as mermaid graphs. The activity diagram contains the following three types of node:
1. Dctivity node. e.g. activity A happens and then activity B happens
```mermaid
graph LR
1["A"] --> 2["B"] 
```

2. Decision node. e.g. activity A happens if Decision D is Yes otherwise activity B happens
```mermaid
graph LR
1{{"D"}} --> |"Yes"| 2["A"]
1 --> |"No"| 3["B"]
```

3. Fork and Join node. e.g. activity A and B happens at the same time after activity F. After A and B finishes, activity J happens
```mermaid
graph LR
1["F"] --> 2["A"]
1 --> 3["B"]
2 --> 4["J"]
3 --> 4
```

Notice that the activity diagram should respect the following constraints:
1. All outgoing edges of a decision node must have a condition
2. There can only be one initial node
3. The activity diagram should be connected
4. The initial node should be able to reach all nodes in the activity diagram

Below are some examples
# Examples
Description:
To confirm the start of delivery (4a), begin by simultaneously completing the following steps: adding missing Master Data, assigning Metering Services, and assigning Meter Operator. After that, notify the customer. Once the registration is confirmed, the process comes to an end.

Activity diagram:
```mermaid
graph LR
0["StartNode"] --> 1["Add missing Master Data"]
0 --> 2["Assign Metering Services"]
0 --> 3["Assign Meter Operator"]
1 --> 4["Notify Costumer"]
2 --> 4
3 --> 4
4 --> 5["registration confirmed"]
```


Description:
In the beginning, the first step is to prepare the import documents. After that, the next step is to obtain the import license. Once the import license is obtained, the goods can be moved to the warehouse. At the same time, it is important to complete the following steps: inspecting the goods and making the necessary payment of duties and taxes. Once these steps are completed, the goods can be released, and that marks the end of the process.

Activity diagram:
```mermaid
graph LR
0["StartNode"] --> 1["Preparing Import Documents"]
1 --> 2["Obtaining Import License"]
2 --> 3["Move Goods to warehouse"]
3 --> 4["Inspection of Goods"]
3 --> 5["Payment of Duties and taxes"]
4 --> 6["Release Goods"]
5 --> 6
6 --> 7["EndNode"]
```


Description:
To rent equipment, start by asking for the equipment. Then, check its availability. If the equipment is available, fill in the borrow form. If it's not available, the process ends. Once the borrow form is filled, sign it. Signing the borrow form requires accessing the data object "Borrow Form". After signing, you can borrow the equipment. If the borrow time ends, return the equipment. Returning the equipment requires checking if it's in good condition. If it is, write the return record. If not, quantify the damage. Writing the return record requires accessing the data object "Borrow Form". If there is any damage, pay for it. After paying, update the inventory. Updating the inventory requires accessing the data object "Inventory". Finally, write the return record and the process ends.


Activity diagram:
```mermaid
graph LR
0["StartNode"] --> 1["Ask for Equipment"]
1 --> 2["Check Availability"]
2 --> 3{{"Equipment available?"}}
3 --> |"Yes"|4["Fill in Borrow Form"]
3 --> |"No"|5["Equipment not available"]
4 --> 6["Sign Borrow Form"]
6 --> 7["Borrow Equipment"]
7 --> 8["Borrow time ended"]
8 --> 9["Return Equipment"]
9 --> 10["Control Equipment State"]
10 --> 11{{"Equipment OK?"}}
11 --> |"Yes"|12["Write Return Record"]
11 --> |"No"|13["Quantify Damage"]
12 --> 14["EndNode"]
13 --> 15["Pay for damage"]
15 --> 16["Update Inventory"]
16 --> 12
``` 


Description:
At the start, AB checks "AC". If "AE" is true, then it proceeds to "AH". However, if "AG" is true, then it goes to "AF". <SEP> Next, it checks "AC" again. If "AE" is true, then it proceeds to "AH". However, if "AG" is true, then it goes to "AF". <SEP> If "AH" is true, then the process ends. If "AF" is true, then it goes to "AD". If "AD" is true, then it goes back to "AH".

Activity diagram:
```mermaid
graph LR
0["StartNode"] --> 1["AB"]
1 --> 2{{"AC"}}
2 --> |"AE"|3["AH"]
2 --> |"AG"|4["AF"]
3 --> 5["EndNode"]
4 --> 6["AD"]
6 --> 3
```


Description:
To begin the process, start by receiving your boarding pass. After that, proceed to the security check. At the same time, make sure to complete the following steps: pass the security screening and pass the luggage screening. Once you have done that, proceed to the departure level. Finally, when you have arrived at the departure level, the process will come to an end.

Activity diagram:
```mermaid
flowchart LR
0["Boarding pass received"] --> 1["Proceed to security check"]
1 --> 2["Pass security screening"]
1 --> 3["Pass luggage screening"]
2 --> 4["Proceed to departure level"]
3 --> 4
4 --> 5["Arrived at departure level"]
```


# Task
First identifying all activities and decisions, then describe how they can be connected, finally output the activity diagram as a mermaid graph:
```mermaid
<result>
```

Description:
{user_input}
"""

PROMPT = """You are an expert software modeler who can create an UML activity diagram from a text description. Given a natural language description, construct an activity diagram by identifying the activities and decisions. Then connect them together to represent the activity flow. The activity diagram are represented as mermaid graphs. The activity diagram contains the following three types of node:
1. Dctivity node. e.g. activity A happens and then activity B happens
```mermaid
graph LR
1["A"] --> 2["B"] 
```

2. Decision node. e.g. activity A happens if Decision D is Yes otherwise activity B happens
```mermaid
graph LR
1{{"D"}} --> |"Yes"| 2["A"]
1 --> |"No"| 3["B"]
```

3. Fork and Join node. e.g. activity A and B happens at the same time after activity F. After A and B finishes, activity J happens
```mermaid
graph LR
1["F"] --> 2["A"]
1 --> 3["B"]
2 --> 4["J"]
3 --> 4
```

Notice that the activity diagram should respect the following constraints:
1. All outgoing edges of a decision node must have a condition
2. There can only be one initial node
3. The activity diagram should be connected
4. The initial node should be able to reach all nodes in the activity diagram


First identifying all activities and decisions, then describe how they can be connected, finally output the activity diagram as a mermaid graph. Make sure to only output one relation per line.
```mermaid
<result>
```
---
To confirm the start of delivery (4a), begin by simultaneously completing the following steps: adding missing Master Data, assigning Metering Services, and assigning Meter Operator. After that, notify the customer. Once the registration is confirmed, the process comes to an end.
---
There are the following activities but no decisions described in the description:
Activities:
0. Add missing Master Data
1. Assign Metering Services
2. Assign Meter Operator
3. Notify Costumer
4. registration confirmed
The process specifies that activities 0, 1, 2 happen at the same time at the beginning, after which activity 3 should happen. Finally, the registration should be confirmed (activity 4). Thus, the final activity diagram should look like the following:
```mermaid
graph LR
0["StartNode"] --> 1["Add missing Master Data"]
0 --> 2["Assign Metering Services"]
0 --> 3["Assign Meter Operator"]
1 --> 4["Notify Costumer"]
2 --> 4
3 --> 4
4 --> 5["registration confirmed"]
```
---
In the beginning, the first step is to prepare the import documents. After that, the next step is to obtain the import license. Once the import license is obtained, the goods can be moved to the warehouse. At the same time, it is important to complete the following steps: inspecting the goods and making the necessary payment of duties and taxes. Once these steps are completed, the goods can be released, and that marks the end of the process.
---
There are the following activities but no decisions described in the description:
Activities
0. Preparing Import Documents
1. Obtaining Import License
2. Move Goods to warehouse
3. Inspection of Goods
4. Payment of Duties and taxes
5. Release Goods

According to the description, activity 0, 1, 2 happen in sequence, then activity 3 and 4 happen a the same time. Finally activity 5 happens. Thus, the final activity diagram should look like the following:
```mermaid
graph LR
0["StartNode"] --> 1["Preparing Import Documents"]
1 --> 2["Obtaining Import License"]
2 --> 3["Move Goods to warehouse"]
3 --> 4["Inspection of Goods"]
3 --> 5["Payment of Duties and taxes"]
4 --> 6["Release Goods"]
5 --> 6
6 --> 7["EndNode"]
```
---
To rent equipment, start by asking for the equipment. Then, check its availability. If the equipment is available, fill in the borrow form. If it's not available, the process ends. Once the borrow form is filled, sign it. Signing the borrow form requires accessing the data object "Borrow Form". After signing, you can borrow the equipment. If the borrow time ends, return the equipment. Returning the equipment requires checking if it's in good condition. If it is, write the return record. If not, quantify the damage. Writing the return record requires accessing the data object "Borrow Form". If there is any damage, pay for it. After paying, update the inventory. Updating the inventory requires accessing the data object "Inventory". Finally, write the return record and the process ends.
---
This description contains the following activities and decisions:
Activities:
0. Ask for Equipment
1. Check Availability
2. Fill in Borrow Form
3. Equipment not available
4. Sign Borrow Form
5. Borrow Equipment
6. Borrow time ended
7. Return Equipment
8. Control Equipment State
9. Write Return Record
10. Quantify Damage
11. Pay for damage
12. Update Inventory
Decisions:
0. Equipment available?
1. Equipment OK?

The description specifies that activities 0, 1 happens in sequence. Then decision 0 is made. If decision 0 is no, then activity 3 happens and then the process ends. if decision 0 is yes, then activities 2, 3, 4, 5, 6, 7 happens in sequence. If decision 1 is no, then activity 10, 11, 12 happens in sequence, followed by activity 12. If decision 1 is yes, then activity 12 happens directly and the process ends. Thus, the final activity diagram should look like the following:
```mermaid
graph LR
0["StartNode"] --> 1["Ask for Equipment"]
1 --> 2["Check Availability"]
2 --> 3{{"Equipment available?"}}
3 --> |"Yes"|4["Fill in Borrow Form"]
3 --> |"No"|5["Equipment not available"]
4 --> 6["Sign Borrow Form"]
6 --> 7["Borrow Equipment"]
7 --> 8["Borrow time ended"]
8 --> 9["Return Equipment"]
9 --> 10["Control Equipment State"]
10 --> 11{{"Equipment OK?"}}
11 --> |"Yes"|12["Write Return Record"]
11 --> |"No"|13["Quantify Damage"]
12 --> 14["EndNode"]
13 --> 15["Pay for damage"]
15 --> 16["Update Inventory"]
16 --> 12
``` 
---
At the start, AB checks "AC". If "AE" is true, then it proceeds to "AH". However, if "AG" is true, then it goes to "AF". <SEP> Next, it checks "AC" again. If "AE" is true, then it proceeds to "AH". However, if "AG" is true, then it goes to "AF". <SEP> If "AH" is true, then the process ends. If "AF" is true, then it goes to "AD". If "AD" is true, then it goes back to "AH".
---
The description contains the following activities and decisions:
Activities:
0. AB
1. AF
2. AD
3. AH
Decisions:
0. AC

The process specifies that activity 0 happens first, followed by decision 0. If decision 0 is AG, then activities 1 and 2 happens, followed by activity 3. If decision 0 is AH, AE, then AH appends directly. Thus, the final activity diagram should look like the following:
```mermaid
graph LR
0["StartNode"] --> 1["AB"]
1 --> 2{{"AC"}}
2 --> |"AE"|3["AH"]
2 --> |"AG"|4["AF"]
3 --> 5["EndNode"]
4 --> 6["AD"]
6 --> 3
```
---
To begin the process, start by receiving your boarding pass. After that, proceed to the security check. At the same time, make sure to complete the following steps: pass the security screening and pass the luggage screening. Once you have done that, proceed to the departure level. Finally, when you have arrived at the departure level, the process will come to an end.
---
The description describes the following activities without decisions:
0. Boarding pass received
1. Proceed to security check
2. Pass security screening
3. Pass luggage screening
4. Proceed to departure level
5. Arrived at departure level

Activity 0 and 1 happen first, followed by activity 2 and 3 happen at the same time. Afterwards, activity 4 and 5 happen. Thus, the final activity diagram should look like the following:
```mermaid
flowchart LR
0["Boarding pass received"] --> 1["Proceed to security check"]
1 --> 2["Pass security screening"]
1 --> 3["Pass luggage screening"]
2 --> 4["Proceed to departure level"]
3 --> 4
4 --> 5["Arrived at departure level"]
```
"""


def get_prompt(prompt_type="fewshot") -> ChatPromptTemplate:
    if prompt_type == "fewshot":
        paragraphs = PROMPT.split("---")
        messages = [("system", paragraphs[0])]
        for i, paragraph in enumerate(paragraphs[1:]):
            if i % 2 == 0:
                messages.append(("user", paragraph))
            else:
                messages.append(("assistant", paragraph))
        messages.append(("human", " {user_input}"))
    elif prompt_type == "simple":
        messages = [("human", PROMPT_SIMPLE_WITH_EXAMPLES)]
    else:
        raise ValueError(f"Unsupported prompt type {prompt_type}")

    return ChatPromptTemplate(messages)
