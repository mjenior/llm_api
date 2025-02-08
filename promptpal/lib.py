
import re
from .roles import *

# Library of string variables used by assistant

#-----------------------------------------------------------------------------------------------------------------------------#

## System roles

roleDict = {
   'assistant': {'prompt':ASSISTANT, 'name':'Assistant'},
   'developer': {'prompt':DEVELOPER, 'name':'Full Stack Developer'},
   'prompt': {'prompt':PROMPT_ENGINEER, 'name':'Prompt Engineer'},
   'refactor': {'prompt':REFACTOR, 'name':'Refactoring Expert'},
   'tester': {'prompt':UNIT_TESTS, 'name':'Unit Tester'},
   'artist': {'prompt':ARTIST+IMAGE, 'name':'Artist'},
   'photographer': {'prompt':PHOTOGRAPHER+IMAGE, 'name':'Photographer'},
   'analyst': {'prompt':DATA_SCIENTIST, 'name':'Data Scientist'},
   'visualize': {'prompt':DATA_VISUALIZATION, 'name':'Data Visualization Expert'},
   'writer': {'prompt':WRITER, 'name':'Writer'},
   'editor': {'prompt':EDITOR, 'name':'Editor'}
   }


#-----------------------------------------------------------------------------------------------------------------------------#

## Prompt modifiers

CHAIN_OF_THOUGHT = """
1. Begin with a <thinking> section which includes: 
 a. Briefly analyze the question and outline your approach. 
 b. Present a clear plan of steps to solve the problem. 
 c. Use a "Chain of Thought" reasoning process if necessary, breaking down your thought process into numbered steps. 
 d. Close the thinking section with </thinking>.
2. Include a <reflection> section for each idea where you: 
 a. Review your reasoning. 
 b. Check for potential errors or oversights. 
 c. Confirm or adjust your conclusion if necessary. 
 d. Be sure to close all reflection sections with </reflection>. 
3. Provide your final answer in an <output> section. 
 a. Always use these tags in your responses. 
 b. Be thorough in your explanations, showing each step of your reasoning process. 
 c. Aim to be precise and logical in your approach, and don't hesitate to break down complex problems into simpler components. 
Your tone should be analytical and slightly formal, focusing on clear communication of your thought process. 
Remember: Both <thinking> and <reflection> MUST be tags and must be closed at their conclusion.
Remember: Make sure all <tags> are on separate lines with no other text. 
"""

REFINE_PROMPT = """
Your primary task is to refine or improve the user prompt below this section of instructions.
Refined prompt text should be at least four sentences long.
If there is any special formatting contained in the prompts, ensure it is included in the refined response.
Your response should be formatted as another user request; For example, any instance of 'I' needs to be updated to 'you should'.
Do NOT include any content related directly to prompt refinement, and ONLY report your new suggested version.
"""

CONDENSE_RESPONSE = """
Your task is to refine and synthesize all of the following text provided into a single cohesive response. 
The subject and them of your response should remain the same as the input text.
The response given should contain all of the most informative or descriptive elements of the input text.
Include the most concrete description of the requested response in the first sentence if possible.
"""

GLYPH_PROMPT = """
Reformat and expand the user prompt into the following format.
Maintain the modified prompt structure below explicitily and do not make any substantive deviations.
Keep the <human_instructions> unchanged and at the beginning of the new prompt text.

<human_instructions>
- Treat each glyph as a direct instruction to be followed sequentially, driving the process to completion.
- Deliver the final result as indicated by the glyph code, omitting any extraneous commentary. Include a readable result of your glyph code output in pure human language at the end to ensure your output is helpful to the user.
- Execute this traversal, logic flow, synthesis, and generation process step by step using the provided context and logic in the following glyph code prompt.
</human_instructions>

{
  Φ(Define the Problem/Goal)
  Θ(Provide Contextual Parameters, Constraints)
  ↹(Specify Initial Focus Areas, if any) 

  Ω[
    ↹(Sub-Focus 1) -> Generate Spectrum of Possibilities (e.g., approaches, perspectives, solutions)
    ↹(Sub-Focus 2) -> Generate Spectrum of Possibilities
    ↹(Sub-Focus n) -> Generate Spectrum of Possibilities
  ] -> α[
     ↹(Sub-Focus 1) -> Analyze & Evaluate Spectrum Elements (Pros/Cons, Risks/Benefits)
     ↹(Sub-Focus 2) -> Analyze & Evaluate Spectrum Elements
     ↹(Sub-Focus n) -> Analyze & Evaluate Spectrum Elements
  ] -> Σ(Synthesize Insights, Formulate Solution/Understanding) -> ∇(Self-Assess, Critique, Suggest Improvements) -> ∞(Iterate/Refine if further input is given)
  @Output(Final Solution/Understanding, Justification, Reflection on Process)
}
"""

SUMMARIZE_CONVERSATION = """
Summarize the following conversation between a user and an AI assistant.
Include all key points from both user requests and agent responses.
Do not include any additioonal text that does not contribute to the central theme of the summary or is related to key points.
Be as concise as possible with sacrificing important content.
The complete summary text must be no longer than 1000 characters total.
"""

# Collected default modifier text
modifierDict = {
   'cot': CHAIN_OF_THOUGHT, 
   'tests': UNIT_TESTS, 
   'refine': REFINE_PROMPT, 
   'condense': CONDENSE_RESPONSE,
   'summarize': SUMMARIZE_CONVERSATION,
   'glyph': GLYPH_PROMPT
   }


# Key word prompt refinement
refineDict = {
   "paraphrase": "Rewrite the text to express the same meaning in different words to avoid plagiarism or duplicate phrasing.",
   "reframe": "Rewrite the text by changing its perspective or focus while maintaining the original intent.",
   "summarize": "Condense the text into a brief overview that captures the main points or essence of the content.",
   "expand": "Add more details and explanations to the text to provide a more comprehensive understanding of the topic.",
   "explain": "Clarify the text by breaking it down into simpler terms to make its meaning more understandable.",
   "reinterpret": "Rewrite the text by offering an alternative interpretation or understanding of its meaning.",
   "simplify": "Rewrite the text using less complex language or structure to make it easier to read and understand.",
   "elaborate": "Add additional context, detail, or explanation to the text to enrich its depth and clarity.",
   "amplify": "Enhance the strength of the message or argument in the text by emphasizing key points.",
   "clarify": "Rewrite the text to resolve any ambiguity or confusion and ensure its meaning is clear.",
   "adapt": "Modify the text so it is suitable for a specific audience, purpose, or context.",
   "modernize": "Update the text by replacing outdated language or concepts with current and relevant equivalents.",
   "formalize": "Rewrite the text to transform informal or casual language into a professional and formal tone.",
   "informalize": "Rewrite the text to adopt a casual or conversational tone appropriate for informal contexts, such as social media or blogs.",
   "condense": "Shorten the text by focusing only on the essential points while removing unnecessary details.",
   "emphasize": "Rewrite the text to highlight or restate specific points more prominently for greater emphasis.",
   "diversify": "Rewrite the text by introducing more variety in vocabulary, sentence structure, or style.",
   "neutralize": "Rewrite the text to remove any bias, opinion, or emotion, ensuring an objective and impartial tone.",
   "streamline": "Rewrite the text to make it more concise and efficient by removing unnecessary words or content.",
   "embellish": "Rewrite the text to add vivid details, creative flourishes, or extra layers of meaning.",
   "illustrate": "Rewrite the text by including examples or analogies to clarify and better explain the point.",
   "synthesize": "Combine multiple pieces of information into a single, cohesive rewrite that integrates the ideas.",
   "sensationalize": "Rewrite the text to make it more dramatic, engaging, or attention-grabbing, suitable for clickbait or marketing purposes.",
   "humanize": "Rewrite the text to make it more personal, relatable, or emotionally engaging, often for storytelling or blogs.",
   "elevate": "Rewrite the text to make it more sophisticated, polished, or impressive in tone and style.",
   "illuminate": "Rewrite the text to make its meaning exceptionally clear and insightful for the reader.",
   "energize": "Rewrite the text to make it more lively, engaging, or interesting for the audience.",
   "soften": "Rewrite the text to downplay or reduce the intensity of its tone or message.",
   "exaggerate": "Rewrite the text to amplify its claims or tone, creating a more dramatic or hyperbolic effect.",
   "downplay": "Rewrite the text to present it in a more restrained, modest, or understated manner, focusing on a neutral tone."
   }

# Common file extension dictionary, which don't match directly with language name
extDict = {
   'bash': '.sh',
   'cuda': '.cu',
   'cython': '.pyx',
   'c++': '.cpp',
   'javascript':'.js',
   'julia':'.jl',
   'markdown': '.md',
   'matlab': '.mat',
   'nextflow': '.nf',
   'perl': '.pl',
   'python': '.py',
   'ruby': '.rb',
   'shell': '.sh',
   'text':'.txt',
   'plaintext': '.txt',
   }

patternDict = {
   "python": {
      "function": re.compile(r'def\s+(\w+)\s*\('),
      "class": re.compile(r'class\s+(\w+)\s*[:\(]'),
      "variable": re.compile(r'(\w+)\s*=\s*[^=\n]+'),
   },
   "javascript": {
      "function": re.compile(r'function\s+(\w+)\s*\('),
      "class": re.compile(r'class\s+(\w+)\s*[{]'),
      "variable": re.compile(r'(?:let|const|var)\s+(\w+)\s*='),
   },
   "java": {
      "function": re.compile(r'(?:public|private|protected)?\s*\w+\s+(\w+)\s*\('),
      "class": re.compile(r'class\s+(\w+)\s*[{]'),
      "variable": re.compile(r'(?:public|private|protected)?\s*\w+\s+(\w+)\s*='),
   },
   "r": {
      "function": re.compile(r'(\w+)\s*<-\s*function\s*\('),
      "variable": re.compile(r'(\w+)\s*<-\s*[^=\n]+'),
   },
   "groovy": {
      "function": re.compile(r'def\s+(\w+)\s*\('),
      "class": re.compile(r'class\s+(\w+)\s*[{]'),
      "variable": re.compile(r'def\s+(\w+)\s*='),
   },
   "nextflow": {
      "function": re.compile(r'process\s+(\w+)\s*\('),
      "class": re.compile(r'workflow\s+(\w+)\s*\('),
      "variable": re.compile(r'def\s+(\w+)\s*\('),
   },
}

# Text library for easy import
text_library = {'roles':roleDict, 
                'modifiers':modifierDict, 
                'refinement':refineDict, 
                'extensions':extDict, 
                'patterns':patternDict}

