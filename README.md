# OpenAI Functions

```
pip install toolcall
```

The most natural way to implement functions for OpenAI tool calling.

`@openai_function`:
- **Argument validation of complex types** using Pydantic `BaseModel` under the hood.
- **Automatic JSON Schema creation** using a mix of docstring parsing, Pydantic's `model_json_schema()`, and custom enhancements.
- **Utility methods for raw tool-call processing**

```py
from toolcall import openai_function
from typing import Literal
import json

@openai_function
def get_stock_price(ticker: str, currency: Literal["USD", "EUR"] = "USD"):
    """
    Get the stock price of a company, by ticker symbol

    Parameters
    ----------
    ticker
        The ticker symbol of the company
    currency
        The currency to use
    """
    return f"182.41 {currency}, -0.48 (0.26%) today"


get_stock_price
```
```json
OpenaiFunction({
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Get the stock price of a company, by ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The ticker symbol of the company"
                },
                "currency": {
                    "type": "string",
                    "description": "The currency to use",
                    "enum": [
                        "USD",
                        "EUR"
                    ],
                    "default": "USD"
                }
            },
            "required": [
                "ticker"
            ]
        }
    }
})
```

Everything you need for implementing function calling is encapsulated in a single object.

## How does it work?

`@openai_function`:
1. Turns your function into a subclass of `pydantic.BaseModel` with your function's parameters as attributes. So, in the example above, running `get_stock_price(ticker="AAPL")` would create an instance of this model, validating the arguments.
2. Creates the JSON schema shown above, and stores it as a class attribute
3. Implements a `.execute()` instance method that passes the instance's attributes to the function you defined.
4. Implements `.run_tool_call()` and `.run_function_call()` class methods that process a raw tool/function call from OpenAI end-to-end, producing a tool message as the result, to send back to OpenAI


## Getting Started

#### Get OpenAI function definition schema
```py
get_stock_price.schema
```
```
{'type': 'function', 'function': {'name': 'get_stock_price', 'description': 'Get the stock price of a company, by ticker symbol', 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string', 'description': 'The ticker symbol of the company'}, 'currency': {'type': 'string', 'description': 'The currency to use', 'enum': ['USD', 'EUR'], 'default': 'USD'}}, 'required': ['ticker']}}}
```

#### Instantiate our pydantic model, validating arguments

```py
validated_function_call = get_stock_price(ticker="AAPL")
```

#### Execute the function, with already-validated arguments
```py
validated_function_call.execute()
```
```
'182.41 USD, -0.48 (0.26%) today'
```

## End-to-End Tool Call Processing, with Error Handling

When an OpenAI model chooses to call the `get_stock_price` function we defined, it sends us a message like this.
```py
message_from_openai = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "type": "function",
            "id": "call_LD0WokrRan5j8B5UehILAdMq",
            "function": {
                "name": "get_stock_price",
                "arguments": "{\"ticker\": \"AAPL\"}"
            },
        }
    ]
}

tool_call = message_from_openai["tool_calls"][0]
```

Our `get_stock_price` has a utility classmethod, `run_tool_call`, to handle this elegantly.

```py
tool_response_message = get_stock_price.run_tool_call(tool_call)
tool_response_message
```
```
{
    'role': 'tool',
    'tool_call_id': 'call_LD0WokrRan5j8B5UehILAdMq',
    'content': '182.41 USD, -0.48 (0.26%) today'
}
```

This method handles all the boilerplate for you:
- Keeping track of the tool call `id`
- Parsing JSON
- Passing arguments to your function model for Pydantic validation
- Executing your function with the validated arguments
- Wrapping the result in a response message

### It also makes error handling easy

The `run_tool_call()` method accepts an `error_handler` argument: a callback function that takes an exception and returns a string to send back to OpenAI, documenting the error.

Pass `error_handler=True` to use the default handler.

Consider this example, where we receive incorrect types:

```py
bad_tool_call = {
    "type": "function",
    "id": "call_LD0WokrRan5j8B5UehILAdMq",
    "function": {
        "name": "get_stock_price",
        "arguments": "{\"ticker\": 5, \"currency\": \"FOOBAR\"}"
    },
}

tool_response_message = get_stock_price.run_tool_call(bad_tool_call, error_handler=True)
tool_response_message
```
```
{
   'role': 'tool',
   'tool_call_id': 'call_LD0WokrRan5j8B5UehILAdMq',
   'content': 
      'Validation failed for the following parameters

      ticker:
        Input: 5
        Error: Input should be a valid string

      currency:
        Input: 'FOOBAR'
        Error: Input should be 'USD' or 'EUR'
      ',
}
```

# Tool Groups

```py
def get_stock_price(ticker: str):
    "Get the stock price of a company, by ticker symbol."
    return "182.41 USD, −0.48 (0.26%) today"

def get_weather(city: str):
    "Get the current weather in a city."
    return "Sunny and 75 degrees"

def get_current_datetime(city: str):
    "Get the current date and time in a city."
    return "Friday, Nov. 10, 2023, 10:00 AM"

group = openai_tool_group([get_stock_price, get_weather, get_current_datetime])
group
```
```
OpenaiToolGroup([
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the stock price of a company, by ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string"
                    }
                },
                "required": [
                    "ticker"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string"
                    }
                },
                "required": [
                    "city"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current date and time in a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string"
                    }
                },
                "required": [
                    "city"
                ]
            }
        }
    }
])
```


> Note: Use `group.tools` to get a list of the raw tool schemas to use in the `tools` argument to OpenAI. Or `group.functions` to get only the functions, for the deprecated function calling API.

#### Now, when we get a function tool call from OpenAI, the group can handle it.
```py
tool_call = {
    "type": "function",
    "id": "call_LD0WokrRan5j8B5UehILAdMq",
    "function": {
        "name": "get_weather",
        "arguments": "{\"city\": \"Denver\"}",
    },
}

tool_response_message = group.run_tool_call(tool_call, error_handler=True)
tool_response_message
```
```
{
    'role': 'tool',
    'tool_call_id': 'call_LD0WokrRan5j8B5UehILAdMq',
    'content': 'Sunny and 75 degrees'
}
```

# Create your own ChatGPT, with automated tool call handling

## Step 1. Conversation handler

```py
import os
import json
from toolcall import openai_tool_group, openai_function, OpenaiToolGroup
from dataclasses import dataclass
from openai import OpenAI
from openai.types.chat.chat_completion import Choice
from typing import Optional


@dataclass
class ChatGPTConversation:
    model: str
    client: OpenAI
    tool_group: OpenaiToolGroup
    messages: list[dict]

    def chat(self):
        result = self.get_openai_response()
        self.add_message(result.message.model_dump(exclude_unset=True))

        if result.message.tool_calls:
            for call in result.message.tool_calls:
                result_msg = self.tool_group.run_tool_call(call, error_handler=True)
                self.add_message(result_msg)

        if result.finish_reason == 'tool_calls':
            self.chat()

    def get_openai_response(self) -> Choice:
        response = self.client.chat.completions.create(
            messages=self.messages,
            model=self.model,
            tools=self.tool_group.tools,
        )
        return response.choices[0]

    def add_message(self, message: dict):
        print(json.dumps(message, indent=4))
        self.messages.append(message)

    def send_message(self, prompt: str):
        self.add_message({"role": "user", "content": prompt})
        self.chat()
```

> Here, `chat()` is a recursive method that continues sending API requests
for as long as the response's `finish_reason='function_call'`.

## Step 2. Create a new conversation
```py
chatgpt = ChatGPTConversation(
    model="gpt-4-1106-preview",
    client=OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    tool_group=group, # using the group defined earlier
    messages=[
        dict(role="system", content="You are a helpful AI assistant."),
    ]
)
```

## Step 4. Exchange messages

**You'll need to use Jupyter notebooks (or interactive terminal) for this**

```py
chatgpt.send_message("Hello, how are you?")
```
```
{
    "role": "user",
    "content": "Hello, how are you?"
}
{
    "role": "assistant",
    "content": "Hello! I'm just a computer program, so I don't have feelings, but I'm ready and functioning properly. How can I assist you today?"
}
```

```py
chatgpt.send_message(
    "I'm enjoying my breakfast here in Denver. Can you list 3 fun "
    "things to do here, then give me a quick morning update?"
)
```
```
{
    "role": "user",
    "content": "I'm enjoying my breakfast here in Denver. Can you list 3 fun things to do here, then give me a quick morning update?"
}
{
    "role": "assistant",
    "content": "Denver is a vibrant city with plenty to offer! Here are three fun things you might enjoy:\n\n1. Explore the Denver Art Museum: The museum is one of the largest in the West and is famous for its collection of American Indian art, as well as its other diverse art collections.\n\n2. Visit the Denver Botanic Gardens: This urban oasis is a great place to enjoy the beauty of nature with a variety of themed gardens, a conservatory, and an amphitheater for seasonal events.\n\n3. Take a stroll through the historic Larimer Square: This historic district is Denver's oldest and most historic block, featuring unique shops, independent boutiques, an energetic nightlife, and some of the city's best restaurants.\n\nNow, let's get you the morning update with the current weather in Denver and the current date and time there. Please hold on a moment.",
    "tool_calls": [
        {
            "id": "call_IvszwiKTgEqxp82BSxt8vyOV",
            "function": {
                "arguments": "{\"city\": \"Denver\"}",
                "name": "get_weather"
            },
            "type": "function"
        },
        {
            "id": "call_oG82zPuqS8V99an5y0hqwdKj",
            "function": {
                "arguments": "{\"city\": \"Denver\"}",
                "name": "get_current_datetime"
            },
            "type": "function"
        }
    ]
}
{
    "role": "tool",
    "tool_call_id": "call_IvszwiKTgEqxp82BSxt8vyOV",
    "content": "Sunny and 75 degrees"
}
{
    "role": "tool",
    "tool_call_id": "call_oG82zPuqS8V99an5y0hqwdKj",
    "content": "Friday, Nov. 10, 2023, 10:00 AM"
}
{
    "role": "assistant",
    "content": "Your morning update for Denver is as follows:\n\n**Weather:** It's currently sunny and 75 degrees, a pleasant morning to enjoy your day!\n\n**Date and Time:** It's Friday, November 10, 2023, and the time is 10:00 AM.\n\nMake the most of your breakfast and have a fantastic time exploring all that Denver has to offer! If you need any more information or assistance, feel free to ask."
}
```

#### If we stitch together the `content` of each *assistant* message (2nd and last messages), we get a continuous block of response text:

> Sure, Denver offers many activities for a fun day out. Here are three fun things to do in Denver:
> 
> 1. **Visit the Denver Art Museum**: Recognized for its collection of American Indian Art and its impressive array of modern and contemporary pieces, the Denver Art Museum is a great place to get a dose of culture. The building itself is an architectural work of art.
> 
> 2. **Explore the Denver Botanic Gardens**: With a wide variety of plants from all corners of the world, as well as a conservatory and sunken amphitheater that hosts various concerts and events, the gardens provide a beautiful and tranquil escape from the city buzz.
> 
> 3. **Take a stroll in the LoDo Historic District**: Lower Downtown, or LoDo as it's affectionately known, is Denver's bustling district filled with late 19th and early 20th-century buildings. You can enjoy boutique shopping, a multitude of restaurants, and a vibrant nightlife.
> 
> For your morning update, let me provide you with the current weather in Denver and the status of the stock market. Just a moment while I gather this information for you.
> 
> Here's your morning update for Denver:
> 
> - **Weather**: It's a sunny day with a current temperature of 75 degrees Fahrenheit.
> - **Stock Market (SPY)**: The SPDR S&P 500 ETF (SPY), a good indicator of the stock market's overall performance, is currently trading at $182.41 USD, with a slight decrease of 0.26% today.
> 
> Enjoy your breakfast and have a fantastic day exploring Denver! If you need any more assistance or information, feel free to ask.


#### This single response was made up of multiple API calls/responses:
1. *Sent*:
    - User prompt
2. *Received*: 
    - Content response (**PART 1**)
    - Tool call to function: `get_weather`
    - Tool call to function: `get_stock_price`
3. *Sent*:
    - Function result from: `get_weather`
    - Function result from: `get_stock_price`
4. *Received*:
    - Content response (**PART 2**)

The response text above combines the content from API responses **2** and **4**.


# Function Calling API

**_(Deprecated by OpenAI)_**

`openai_function` and `OpenaiToolGroup` also support the deprecated function-calling API. 

- The function schema is accessible by `schema["function"]`, which can be passed inside a list to the `functions` argument in your OpenAI request. In an `OpenaiToolGroup`, this list is quickly accessible via the `functions` property.
- Use `.run_function_call()` instead of `.run_tool_call()`, to process results.

Example:

```py
message_from_openai = {
    "role": "assistant",
    "content": None,
    "function_call": {
        "name": "get_stock_price",
        "arguments": "{\"ticker\": \"AAPL\"}",
    },
}
function_call = message_from_openai["function_call"]
```

Use `run_function_call()`

```py
function_response_message = get_stock_price.run_function_call(function_call)
function_response_message
```
```
{
    'role': 'function',
    'name': 'get_stock_price',
    'content': '182.41 USD, -0.48 (0.26%) today'
}
```

## Create your own ChatGPT (function calling version)

### Conversation handler

```py
# subclassing our previous version

class ChatGPTFunctionConversation(ChatGPTConversation):

    def get_openai_response(self) -> Choice:
        response = self.client.chat.completions.create(
            messages=self.messages,
            model=self.model,
            functions=self.tool_group.functions,  # Pass `functions` instead of `tools`
        )
        return response.choices[0]

    def chat(self):
        result = self.get_openai_response()
        self.add_message(result.message.model_dump(exclude_unset=True))

        # Handling just one function call, rather than a list of tool calls.
        func_call = result.message.function_call
        if func_call:
            result_msg = self.tool_group.run_function_call(func_call, error_handler=True)
            self.add_message(result_msg)

        if result.finish_reason == 'function_call':
            self.chat()
```

Using the same configuration as before, and sending the same messages.
```py
chatgpt = ChatGPTConversation(
    model="gpt-4-1106-preview",
    client=OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    tool_group=group,
    messages=[
        dict(role="system", content="You are a helpful AI assistant."),
    ]
)
```

```py
chatgpt.send_message("Hello, how are you?")
```
```
{
    "role": "user",
    "content": "Hello, how are you?"
}
{
    "content": "Hello! I'm just a computer program, so I don't have feelings, but I'm functioning optimally and ready to assist you. How can I help you today?",
    "role": "assistant"
}
```

```py
chatgpt.send_message(
    "I'm enjoying my breakfast here in Denver. Can you list 3 fun "
    "things to do here, then give me a quick morning update?"
)
```
```
{
    "role": "user",
    "content": "I'm enjoying my breakfast here in Denver. Can you list 3 fun things to do here, then give me a quick morning update?"
}
{
    "role": "assistant",
    "content": "Denver offers a wide variety of activities to enjoy. Here are three fun things you could consider doing:\n\n1. **Visit the Denver Art Museum**: The museum is one of the largest art museums between the West Coast and Chicago and offers a diverse collection of artworks from around the world.\n\n2. **Explore the Denver Botanic Gardens**: With a variety of themed gardens, a conservatory, and an amphitheater for summer concerts, the Denver Botanic Gardens provides a beautiful and relaxing urban oasis.\n\n3. **Walk or Bike the Cherry Creek Trail**: This scenic path runs through the heart of Denver and is great for biking, walking, or even just taking a leisurely stroll. It will give you great views of the city and nature alike.\n\nFor your morning update, let's check the current weather in Denver and see what the day looks like for you. I'll also provide you with the current date and time to start your day off informed. Please give me a moment to gather that information for you.",
    "function_call": {
        "name": "get_weather",
        "arguments": "{\"city\":\"Denver\"}"
    }
}
{
    "role": "function",
    "name": "get_weather",
    "content": "Sunny and 75 degrees"
}
{
    "role": "assistant",
    "content": null,
    "function_call": {
        "arguments": "{\"city\":\"Denver\"}",
        "name": "get_current_datetime"
    }
}
{
    "role": "function",
    "name": "get_current_datetime",
    "content": "Friday, Nov. 10, 2023, 10:00 AM"
}
{
    "role": "assistant",
    "content": "Here's your morning update for Denver:\n\n- **Weather**: It's a beautiful sunny day with temperatures around 75 degrees Fahrenheit. Perfect weather for any outdoor activities!\n- **Date and Time**: It's currently Friday, November 10, 2023, at 10:00 AM.\n\nWhether you decide to spend the day indoors or outdoors, it looks like it's shaping up to be a lovely day in Denver. Enjoy your breakfast and have a great day!"
}
```
