
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
### Task
Your primary objective is to refine or improve the user prompt provided below.
Your refinement should enhance clarity, specificity, and completeness to maximize the likelihood of a high-quality response.

### Requirements
- The refined prompt must be at least **four sentences long**.
- Maintain any **special formatting** (e.g., bullet points, code blocks) from the original prompt.
- Ensure the refined prompt is written as a **user request** (e.g., replace 'I' with 'You should').
- Do **not** include any meta-commentary on prompt refinement—your response should only contain the improved prompt.

### Output Format
Your response should consist **only of the refined prompt**, without any additional explanation or formatting beyond what is necessary for clarity.
"""

CONDENSE_RESPONSE = """
### Task
Your objective is to **synthesize and refine** the provided text into a **single, cohesive response** that retains the original subject and theme.

### Requirements
- Extract and integrate the **most informative and descriptive elements** from the input text.
- Ensure the response is **concise yet comprehensive**, avoiding redundancy while preserving key details.
- If possible, **begin with the most concrete and precise description** of the topic to establish clarity upfront.
- Maintain logical flow and coherence, ensuring the response reads naturally as a unified piece rather than a collection of excerpts.

### Output Format
Your response should consist **only of the synthesized text**, without additional commentary or formatting beyond what enhances readability.
"""

GLYPH_PROMPT = """
Reformat and expand the user prompt into the following format.
Maintain the modified prompt structure below explicitily and do not make any substantive deviations.
Keep the <human_instructions> unchanged and at the beginning of the new prompt text.

<human_instructions>
- Treat each glyph as a direct instruction to be followed sequentially, driving the process to completion.
- Deliver the final result as indicated by the glyph code, omitting any extraneous commentary. 
- Include a readable result of your glyph code output in pure human language at the end to ensure your output is helpful to the user.
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
### Task  
Summarize the following conversation between a **user** and an **AI assistant**, preserving all key points from both user requests and assistant responses.

### Requirements
- Capture the **essential details** and main points from both sides of the conversation.
- Exclude any information that does not contribute to the **central theme** or key takeaways.
- Maintain **conciseness** without sacrificing important content or clarity.
- The summary **must not exceed 10,000 characters** in total.

### Output Format  
Your response should consist **only of the summarized conversation**, structured in a clear and logical manner without additional commentary.
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


# Key word prompt refinement, expand, amplify
refineDict = {
   "paraphrase": "Rewrite the text using different words while keeping the original meaning.",
   "reframe": "Change the perspective or focus of the text while maintaining its intent.",
   "summarize": "Condense the text to highlight key points.",
   "expand": "Add details and explanations for a more comprehensive understanding.",
   "explain": "Clarify the text by simplifying its meaning.",
   "reinterpret": "Offer an alternative understanding of the text.",
   "simplify": "Use less complex language for easier comprehension.",
   "elaborate": "Provide additional context and details to enrich clarity.",
   "amplify": "Emphasize key points to strengthen the message.",
   "clarify": "Remove ambiguity to ensure clear meaning.",
   "adapt": "Modify the text for a specific audience, purpose, or context.",
   "modernize": "Update outdated language or concepts with current equivalents.",
   "formalize": "Transform casual language into a professional tone.",
   "informalize": "Adjust the text to a conversational style.",
   "condense": "Shorten the text while preserving essential points.",
   "emphasize": "Highlight specific points more prominently.",
   "diversify": "Vary vocabulary, sentence structure, or style.",
   "neutralize": "Remove bias or emotion for an objective tone.",
   "streamline": "Make the text more concise and efficient.",
   "embellish": "Add vivid details or creative flourishes.",
   "illustrate": "Include examples or analogies for clarity.",
   "synthesize": "Combine multiple ideas into a cohesive rewrite.",
   "sensationalize": "Make the text more dramatic and engaging.",
   "humanize": "Make the text more personal and relatable.",
   "elevate": "Refine the text to be more polished and sophisticated.",
   "energize": "Make the text more lively and engaging.",
   "soften": "Reduce intensity for a gentler tone.",
   "exaggerate": "Amplify claims or tone for dramatic effect.",
   "downplay": "Present in a more restrained and understated manner."
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
