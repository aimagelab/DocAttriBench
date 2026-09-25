MODEL_TEMPLATES = dict()


#----------------
# ANSWER BOX
#----------------

# The answerBox template is used to generate both the answer and the bounding box


MODEL_TEMPLATES["answerBox"] = dict()


MODEL_TEMPLATES["answerBox"]["Qwen_Qwen2.5-VL-7B-Instruct-AWQ"] = {
    "system_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to answer the user's query based on the provided image and the context.
When you answer, provide evidence bounding boxes in the format <box> x1 y1 x2 y2 </box> where (x1, y1) is the top-left corner.
Add an evidence bounding box at the end of each generated sentence.

If you cannot find the answer, respond with "I don't know".

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.""",
    "user_template": """{query}""",
    "image_args": [
        "resized_height",
        "resized_width",
    ],
}


MODEL_TEMPLATES["answerBox"]["Qwen_Qwen2.5-VL-32B-Instruct-AWQ"] = {
    "system_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to answer the user's query based on the provided image and the context.
When you answer, provide evidence bounding boxes in the format <box> x1 y1 x2 y2 </box> where (x1, y1) is the top-left corner.
Add an evidence bounding box at the end of each generated sentence.

If you cannot find the answer, respond with "I don't know".

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.""",
    "user_template": """{query}""",
    "image_args": [
        "resized_height",
        "resized_width",
    ],
}


MODEL_TEMPLATES["answerBox"]["Qwen_Qwen2.5-VL-3B-Instruct"] = {
    "system_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to answer the user's query based on the provided image and the context.
When you answer, provide evidence bounding boxes in the format <box> x1 y1 x2 y2 </box> where (x1, y1) is the top-left corner.
Add an evidence bounding box at the end of each generated sentence.

If you cannot find the answer, respond with "I don't know".

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.""",
    "user_template": """{query}""",
    "image_args": [
        "resized_height",
        "resized_width",
    ],
}


MODEL_TEMPLATES["answerBox"]["Qwen_Qwen2.5-VL-7B-Instruct"] = {
    "system_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to answer the user's query based on the provided image and the context.
When you answer, provide evidence bounding boxes in the format <box> x1 y1 x2 y2 </box> where (x1, y1) is the top-left corner.
Add an evidence bounding box at the end of each generated sentence.

If you cannot find the answer, respond with "I don't know".

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.""",
    "user_template": """{query}""",
    "image_args": [
        "resized_height",
        "resized_width",
    ],
}


MODEL_TEMPLATES["answerBox"]["Qwen_Qwen3-VL-2B-Instruct"] = {
    "system_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to answer the user's query based on the provided image and the context.
When you answer, provide evidence bounding boxes in the format <box> x1 y1 x2 y2 </box> where (x1, y1) is the top-left corner.
Add an evidence bounding box at the end of each generated sentence.

If you cannot find the answer, respond with "I don't know".

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.""",
    "user_template": """{query}""",
    "image_args": [
        "resized_height",
        "resized_width",
    ],
}


MODEL_TEMPLATES["answerBox"]["Qwen_Qwen3-VL-8B-Instruct"] = {
    "system_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to answer the user's query based on the provided image and the context.
When you answer, provide evidence bounding boxes in the format <box> x1 y1 x2 y2 </box> where (x1, y1) is the top-left corner.
Add an evidence bounding box at the end of each generated sentence.

If you cannot find the answer, respond with "I don't know".

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.""",
    "user_template": """{query}""",
    "image_args": [
        "resized_height",
        "resized_width",
    ],
}


MODEL_TEMPLATES["answerBox"]["Qwen_Qwen3-VL-32B-Instruct-FP8"] = {
    "system_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to answer the user's query based on the provided image and the context.
When you answer, provide evidence bounding boxes in the format <box> x1 y1 x2 y2 </box> where (x1, y1) is the top-left corner.
Add an evidence bounding box at the end of each generated sentence.

If you cannot find the answer, respond with "I don't know".

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.""",
    "user_template": """{query}""",
    "image_args": [
        "resized_height",
        "resized_width",
    ],
}


MODEL_TEMPLATES["answerBox"]["Qwen_Qwen3-VL-32B-Instruct"] = {
    "system_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to answer the user's query based on the provided image and the context.
When you answer, provide evidence bounding boxes in the format <box> x1 y1 x2 y2 </box> where (x1, y1) is the top-left corner.
Add an evidence bounding box at the end of each generated sentence.

If you cannot find the answer, respond with "I don't know".

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.""",
    "user_template": """{query}""",
    "image_args": [
        "resized_height",
        "resized_width",
    ],
}


MODEL_TEMPLATES["answerBox"]["OpenGVLab_InternVL2_5-2B"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to answer the user's query based on the provided image and the context.
When you answer, provide evidence bounding boxes in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.
Add an evidence bounding box at the end of each generated sentence.

If you cannot find the answer, respond with "I don't know".

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}""",
}


MODEL_TEMPLATES["answerBox"]["OpenGVLab_InternVL2_5-8B"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to answer the user's query based on the provided image and the context.
When you answer, provide evidence bounding boxes in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.
Add an evidence bounding box at the end of each generated sentence.

If you cannot find the answer, respond with "I don't know".

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}""",
}


MODEL_TEMPLATES["answerBox"]["OpenGVLab_InternVL2_5-38B"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to answer the user's query based on the provided image and the context.
When you answer, provide evidence bounding boxes in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.
Add an evidence bounding box at the end of each generated sentence.

If you cannot find the answer, respond with "I don't know".

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}""",
}


MODEL_TEMPLATES["answerBox"]["OpenGVLab_InternVL3-2B-Instruct"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to answer the user's query based on the provided image and the context.
When you answer, provide evidence bounding boxes in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.
Add an evidence bounding box at the end of each generated sentence.

If you cannot find the answer, respond with "I don't know".

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}""",
}


MODEL_TEMPLATES["answerBox"]["OpenGVLab_InternVL3-8B-Instruct"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to answer the user's query based on the provided image and the context.
When you answer, provide evidence bounding boxes in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.
Add an evidence bounding box at the end of each generated sentence.

If you cannot find the answer, respond with "I don't know".

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}""",
}


MODEL_TEMPLATES["answerBox"]["OpenGVLab_InternVL3-38B-Instruct"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to answer the user's query based on the provided image and the context.
When you answer, provide evidence bounding boxes in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.
Add an evidence bounding box at the end of each generated sentence.

If you cannot find the answer, respond with "I don't know".

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}""",
}


MODEL_TEMPLATES["answerBox"]["OpenGVLab_InternVL3_5-2B-Instruct"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to answer the user's query based on the provided image and the context.
When you answer, provide evidence bounding boxes in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.
Add an evidence bounding box at the end of each generated sentence.

If you cannot find the answer, respond with "I don't know".

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}""",
}


MODEL_TEMPLATES["answerBox"]["OpenGVLab_InternVL3_5-8B-Instruct"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to answer the user's query based on the provided image and the context.
When you answer, provide evidence bounding boxes in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.
Add an evidence bounding box at the end of each generated sentence.

If you cannot find the answer, respond with "I don't know".

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}""",
}


MODEL_TEMPLATES["answerBox"]["OpenGVLab_InternVL3_5-38B-Instruct"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to answer the user's query based on the provided image and the context.
When you answer, provide evidence bounding boxes in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.
Add an evidence bounding box at the end of each generated sentence.

If you cannot find the answer, respond with "I don't know".

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}""",
}


#----------------
# BOX
#----------------

# The box template is used to generate only the evidence box


MODEL_TEMPLATES["box"] = dict()


MODEL_TEMPLATES["box"]["Qwen_Qwen2.5-VL-7B-Instruct-AWQ"] = {
    "system_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the answer to the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format <box> x1 y1 x2 y2 </box> where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.""",
    "user_template": """{query}""",
    "image_args": [
        "resized_height",
        "resized_width",
    ],
}


MODEL_TEMPLATES["box"]["Qwen_Qwen2.5-VL-32B-Instruct-AWQ"] = {
    "system_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the answer to the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format <box> x1 y1 x2 y2 </box> where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.""",
    "user_template": """{query}""",
    "image_args": [
        "resized_height",
        "resized_width",
    ],
}


MODEL_TEMPLATES["box"]["Qwen_Qwen2.5-VL-3B-Instruct"] = {
    "system_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the answer to the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format <box> x1 y1 x2 y2 </box> where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.""",
    "user_template": """{query}""",
    "image_args": [
        "resized_height",
        "resized_width",
    ],
}


MODEL_TEMPLATES["box"]["Qwen_Qwen2.5-VL-7B-Instruct"] = {
    "system_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the answer to the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format <box> x1 y1 x2 y2 </box> where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.""",
    "user_template": """{query}""",
    "image_args": [
        "resized_height",
        "resized_width",
    ],
}


MODEL_TEMPLATES["box"]["Qwen_Qwen3-VL-2B-Instruct"] = {
    "system_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the answer to the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format <box> x1 y1 x2 y2 </box> where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.""",
    "user_template": """{query}""",
    "image_args": [
        "resized_height",
        "resized_width",
    ],
}


MODEL_TEMPLATES["box"]["Qwen_Qwen3-VL-8B-Instruct"] = {
    "system_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the answer to the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format <box> x1 y1 x2 y2 </box> where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.""",
    "user_template": """{query}""",
    "image_args": [
        "resized_height",
        "resized_width",
    ],
}


MODEL_TEMPLATES["box"]["Qwen_Qwen3-VL-32B-Instruct-FP8"] = {
    "system_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the answer to the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format <box> x1 y1 x2 y2 </box> where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.""",
    "user_template": """{query}""",
    "image_args": [
        "resized_height",
        "resized_width",
    ],
}


MODEL_TEMPLATES["box"]["OpenGVLab_InternVL2_5-2B"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the answer to the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box []

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}"""
}


MODEL_TEMPLATES["box"]["OpenGVLab_InternVL2_5-8B"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the answer to the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box []

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}"""
}


MODEL_TEMPLATES["box"]["OpenGVLab_InternVL2_5-38B"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the answer to the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box []

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}"""
}


MODEL_TEMPLATES["box"]["OpenGVLab_InternVL3-2B-Instruct"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the answer to the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box []

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}"""
}


MODEL_TEMPLATES["box"]["OpenGVLab_InternVL3-8B-Instruct"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the answer to the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box []

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}"""
}


MODEL_TEMPLATES["box"]["OpenGVLab_InternVL3-38B-Instruct"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the answer to the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box []

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}"""
}


MODEL_TEMPLATES["box"]["OpenGVLab_InternVL3_5-2B-Instruct"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the answer to the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box []

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}"""
}


MODEL_TEMPLATES["box"]["OpenGVLab_InternVL3_5-8B-Instruct"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the answer to the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box []

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}"""
}


MODEL_TEMPLATES["box"]["OpenGVLab_InternVL3_5-38B-Instruct"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the answer to the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box []

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}"""
}



#----------------
# BOX FROM ANSWER
#----------------

# The boxFromAnswer template is used to generate the bounding box when the answer is given.


MODEL_TEMPLATES["boxFromAnswer"] = dict()


MODEL_TEMPLATES["boxFromAnswer"]["Qwen_Qwen2.5-VL-7B-Instruct-AWQ"] = {
    "system_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the evidence used to answer the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format <box> x1 y1 x2 y2 </box> where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.""",
    "user_template": """Query:
{query}

Answer:
{answer}""",
    "image_args": [
        "resized_height",
        "resized_width",
    ],
}



MODEL_TEMPLATES["boxFromAnswer"]["Qwen_Qwen2.5-VL-32B-Instruct-AWQ"] = {
    "system_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the evidence used to answer the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format <box> x1 y1 x2 y2 </box> where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.""",
    "user_template": """Query:
{query}

Answer:
{answer}""",
    "image_args": [
        "resized_height",
        "resized_width",
    ],
}


MODEL_TEMPLATES["boxFromAnswer"]["Qwen_Qwen2.5-VL-3B-Instruct"] = {
    "system_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the evidence used to answer the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format <box> x1 y1 x2 y2 </box> where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.""",
    "user_template": """Query:
{query}

Answer:
{answer}""",
    "image_args": [
        "resized_height",
        "resized_width",
    ],
}


MODEL_TEMPLATES["boxFromAnswer"]["Qwen_Qwen2.5-VL-7B-Instruct"] = {
    "system_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the evidence used to answer the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format <box> x1 y1 x2 y2 </box> where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.""",
    "user_template": """Query:
{query}

Answer:
{answer}""",
    "image_args": [
        "resized_height",
        "resized_width",
    ],
}



MODEL_TEMPLATES["boxFromAnswer"]["Qwen_Qwen3-VL-2B-Instruct"] = {
    "system_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the evidence used to answer the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format <box> x1 y1 x2 y2 </box> where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.""",
    "user_template": """Query:
{query}

Answer:
{answer}""",
    "image_args": [
        "resized_height",
        "resized_width",
    ],
}


MODEL_TEMPLATES["boxFromAnswer"]["Qwen_Qwen3-VL-8B-Instruct"] = {
    "system_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the evidence used to answer the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format <box> x1 y1 x2 y2 </box> where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.""",
    "user_template": """Query:
{query}

Answer:
{answer}""",
    "image_args": [
        "resized_height",
        "resized_width",
    ],
}


MODEL_TEMPLATES["boxFromAnswer"]["Qwen_Qwen3-VL-32B-Instruct-FP8"] = {
    "system_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the evidence used to answer the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format <box> x1 y1 x2 y2 </box> where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.""",
    "user_template": """Query:
{query}

Answer:
{answer}""",
    "image_args": [
        "resized_height",
        "resized_width",
    ],
}


MODEL_TEMPLATES["boxFromAnswer"]["OpenGVLab_InternVL2_5-2B"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the evidence used to answer the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}

Answer:
{answer}""",
}


MODEL_TEMPLATES["boxFromAnswer"]["OpenGVLab_InternVL2_5-8B"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the evidence used to answer the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}

Answer:
{answer}""",
}


MODEL_TEMPLATES["boxFromAnswer"]["OpenGVLab_InternVL2_5-38B"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the evidence used to answer the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}

Answer:
{answer}""",
}


MODEL_TEMPLATES["boxFromAnswer"]["OpenGVLab_InternVL3-2B-Instruct"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the evidence used to answer the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}

Answer:
{answer}""",
}


MODEL_TEMPLATES["boxFromAnswer"]["OpenGVLab_InternVL3-8B-Instruct"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the evidence used to answer the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}

Answer:
{answer}""",
}


MODEL_TEMPLATES["boxFromAnswer"]["OpenGVLab_InternVL3-38B-Instruct"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the evidence used to answer the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}

Answer:
{answer}""",
}


MODEL_TEMPLATES["boxFromAnswer"]["OpenGVLab_InternVL3_5-2B-Instruct"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the evidence used to answer the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}

Answer:
{answer}""",
}


MODEL_TEMPLATES["boxFromAnswer"]["OpenGVLab_InternVL3_5-8B-Instruct"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the evidence used to answer the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}

Answer:
{answer}""",
}


MODEL_TEMPLATES["boxFromAnswer"]["OpenGVLab_InternVL3_5-38B-Instruct"] = {
    "user_template": """You are an agent excelent at identifying evidence in a document page.
Your job is to locate the evidence used to answer the user's query on the provided image.
When you answer, provide only the evidence bounding box in the format [x1,y1,x2,y2] where (x1, y1) is the top-left corner.

If you cannot find the answer, respond with the empty box <box> </box>.

If the evidence is part of a paragraph, return the bounding box of the entire paragraph.
If the evidence is part of a table, return the bounding box of the entire table.
If the evidence is part of a figure, return the bounding box of the entire figure.

User's query:
{query}

Answer:
{answer}""",
}
