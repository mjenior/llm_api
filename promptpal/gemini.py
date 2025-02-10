
from propmptpal.core import CreateAgent
from google import genai
from google.genai import types


# Confirm environment API key
api_key = os.getenv("GEMINI_API_KEY")
if api_key is None:
    raise EnvironmentError("GEMINI_API_KEY environment variable not found!")

# Initialize Google client and conversation chat
client = genai.Client(api_key=api_key)
chat = client.chats.create(model="gemini-2.0-flash")
chat.current_chat_calls = 0
client.chat_ids = set([chat.id])


# Google sub-class
class Gemini(CreateAgent):
	    """
    A handler for managing queries to the Google GenAI API, including prompt preparation,
    API request submission, response processing, and logging.

    This class provides a flexible interface to interact with Googles models (e.g., Gemini-2.0)
    It supports features such as associative prompt refinement, chain-of-thought reasoning, 
    code extraction, logging, unit testing, and much more.
    """
    def __init__(self, 
        self,
        model="gemini-2.0-flash",
        threshold=0.55,
        **kwargs,  # Pass remaining arguments to the base class
    ):
        # Set default valid models for Gemini
        google_models = ["gemini-2.0-flash", 
            "gemini-2.0-flash-lite", 
            "gemini-1.5-flash"]
        # As of February 8, 2025
        google_rates = {
           "gemini-2.0-flash": (0.1, 0.4),
           "gemini-2.0-flash-lite": (0.075, 0.3),
           "gemini-1.5-flash": (0.075, 0.3),
        }


        # Initialize the base class with all parameters
        super().__init__(model=model, valid_models=google_models, **kwargs)
        self.self.small_model = 'gemini-2.0-flash-lite'

        # Gemini-specific attributes
        self.threshold = threshold

    	# Agent-specific chat params
        global chat
        self.chat_id = chat.id
        self.chat_history = []
        chat.message_limit = message_limit
        if self.new_chat == True:
            self.start_new_chat()

        # Update token counters
        global total_tokens
        if self.model not in total_tokens.keys():
            total_tokens[self.model] = {"prompt": 0, "completion": 0}

        # Generate agent configuration
        self._create_agent_configuration()

    def _init_chat_completion(self, prompt, model):
        """Initialize and submit a single chat completion request"""

    	cfg = types.GenerateContentConfig(
            temperature=self.temperature,
    		topP=self.top_p,
    		seed=self.seed,
    		candidateCount=self.iterations)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        completion = client.models.generate_content(model=model, contents=prompt, config=cfg)
		
        self._update_token_count(completion)
        self._calculate_cost(google_rates)

        return completion.text

    def start_new_chat(self, context=''):
        """Start a new chat with only the current agent and adds previous context if needed."""
        global chat
        chat = client.chats.create(model=self.model, context=context)
        chat.current_chat_calls = 0
        chat.message_limit = self.message_limit

        global client
        client.chat_ids |= set([chat.id])
        self.chat_id = chat.id

        # Report
        self._log_and_print(f"New chat created and added to current agent: {self.chat_id}\n", 
            self.verbose, self.logging)

    def _update_token_count(self, response_obj):
        """Updates token count for prompt and completion."""
        usage = response_obj.usage_metadata

        global total_tokens
        total_tokens[self.model]["prompt"] += usage['prompt_token_count']
        total_tokens[self.model]["completion"] += usage['candidates_token_count']
        # Agent-specific counts
        self.tokens["prompt"] += usage['prompt_token_count']
        self.tokens["completion"] += usage['candidates_token_count']





    # Update for google
    def _get_current_messages(self):
        """Fetches all messages from a thread in order and returns them as a text block."""

        messages = client.beta.threads.messages.list(chat_id=self.chat_id)
        sorted_messages = sorted(messages.data, key=lambda msg: msg.created_at)


        conversation = []
        for entry in self.chat_history:

            conversation.append()

        return "\n\n".join(conversation)

    


    def _send_chat_message(self) -> str:
        """Sends a user prompt to an existing chat and retrieves the response."""
        self.chat_history.append({"role": "user", "content": self.prompt})

        # Send user prompt to existing chat.
        response = chat.send_message(self.prompt, config=self.agent)

        # Add the model's response to the chat history
        chat_history.append({"role": "model", "content": response.text})
		
		return response.text

	def _create_agent_configuration(self):
        """
        Creates a new assistant based on user-defined parameters

        Returns:
            New assistant config
        """
        self.agent = types.GenerateContentConfig(
        		system_instruction=self.role,
        		temperature=self.temperature,
        		topP=self.top_p,
        		seed=self.seed,
        		candidateCount=self.iterations,
        		tools=[types.Tool(google_search=types.GoogleSearchRetrieval(
	        		dynamic_retrieval_config=types.DynamicRetrievalConfig(
	        			dynamic_threshold=self.threshold)))])
