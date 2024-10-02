---
title: "Conversational AI"
subtitle: "Real-Time Voice Bot with Deepgram, Groq, and LangChain"
date: "07-2024"
---

<p align="center">[GitHub](https://github.com/karthikyeluripati/chatbot_v1)</p>

In the rapidly evolving landscape of conversational AI, voice-enabled bots are becoming increasingly sophisticated and accessible. Today, I'm excited to share a project that demonstrates the power of combining cutting-edge speech recognition, natural language processing, and text-to-speech technologies to create a responsive and engaging voice bot.

## Project Overview: Quick Voice Bot Demo

This alpha demo showcases a voice bot capable of engaging in real-time conversations with users. By leveraging streaming capabilities for both speech-to-text (STT) and text-to-speech (TTS), along with state-of-the-art language models, we've created a system that feels remarkably natural and responsive.

### Key Features:

1. **Real-time Speech Recognition**: Utilizing Deepgram's advanced STT capabilities.
2. **Natural Language Understanding**: Powered by Groq's language models through LangChain.
3. **Dynamic Text-to-Speech**: Deepgram's TTS service for generating human-like responses.
4. **Streaming Pipelines**: Implemented for both STT and TTS to minimize latency.
5. **Flexible Language Model Integration**: Easy switching between different LLMs (e.g., Groq, OpenAI).

## Technical Deep Dive

Let's explore the core components that make this voice bot possible:

### Speech-to-Text with Deepgram

The `TranscriptCollector` and `get_transcript` function work together to capture and process incoming audio:

```python
class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

async def get_transcript(callback):
    # ... (Deepgram setup code)
    
    async def on_message(self, result, **kwargs):
        sentence = result.channel.alternatives[0].transcript
        
        if not result.speech_final:
            transcript_collector.add_part(sentence)
        else:
            # Process final sentence
            transcript_collector.add_part(sentence)
            full_sentence = transcript_collector.get_full_transcript()
            if len(full_sentence.strip()) > 0:
                print(f"Human: {full_sentence}")
                callback(full_sentence)
                transcript_collector.reset()
                transcription_complete.set()
```

This setup allows for real-time transcription, collecting partial results and triggering actions when a complete sentence is detected.

### Language Model Processing with LangChain

The `LanguageModelProcessor` class integrates LangChain to manage conversations with the chosen language model:

```python
class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Load system prompt and set up conversation chain
        # ...

    def process(self, text):
        self.memory.chat_memory.add_user_message(text)
        response = self.conversation.invoke({"text": text})
        self.memory.chat_memory.add_ai_message(response['text'])
        return response['text']
```

This class handles the conversation flow, maintaining context through `ConversationBufferMemory` and generating appropriate responses based on the input text.

### Text-to-Speech with Deepgram

The `TextToSpeech` class manages the conversion of text responses to speech:

```python
class TextToSpeech:
    DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    MODEL_NAME = "aura-helios-en"

    def speak(self, text):
        # ... (Deepgram API setup)

        with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    # Process and play audio chunks
                    # ...
```

This implementation uses streaming to start playing audio as soon as the first chunks are received, significantly reducing perceived latency.

### Bringing It All Together

The `ConversationManager` class orchestrates the entire process:

```python
class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()

    async def main(self):
        while True:
            await get_transcript(self.handle_full_sentence)
            
            if "goodbye" in self.transcription_response.lower():
                break
            
            llm_response = self.llm.process(self.transcription_response)

            tts = TextToSpeech()
            tts.speak(llm_response)

            self.transcription_response = ""
```

This main loop continuously listens for user input, processes it through the language model, and generates spoken responses until the user says "goodbye".

## Potential Applications and Future Enhancements

This voice bot demo serves as a foundation for numerous exciting applications:

1. **Customer Service**: Deploying intelligent voice assistants for 24/7 support.
2. **Educational Tools**: Creating interactive tutoring systems for various subjects.
3. **Accessibility Solutions**: Developing voice-controlled interfaces for individuals with limited mobility.
4. **Virtual Companions**: Building empathetic AI companions for elderly care or mental health support.

Future enhancements could include:

- Multi-language support
- Emotion detection and responsive tone adjustment
- Integration with external knowledge bases for more informed responses
- Voice cloning capabilities for personalized TTS voices

## Conclusion

The Quick Voice Bot Demo showcases the potential of combining real-time speech processing with advanced language models. By leveraging the strengths of Deepgram, Groq, and LangChain, we've created a system that hints at the future of human-AI interaction.

As we continue to refine and expand this technology, the possibilities for creating more natural, responsive, and helpful AI assistants are truly exciting. Whether you're a developer looking to build on this foundation or an enthusiast curious about the future of conversational AI, I hope this project inspires you to explore the incredible potential of voice-enabled AI systems.

Feel free to check out the full source code and contribute to the project on GitHub. Let's push the boundaries of what's possible in the world of voice-enabled AI together!

