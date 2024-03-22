"use client";

import { useRef, useState } from "react";
import { MessageProvider, useMessageContext } from "@/app/message-context";
import { chain } from "@/client";

export default function Home() {
  return (
    <MessageProvider>
      <Main />
    </MessageProvider>
  );
}

function isStatusCode(
  chunk: unknown
): chunk is { message: string; status_code: number } {
  return typeof chunk === "object" && chunk !== null && "status_code" in chunk;
}

function isAnswer(chunk: unknown): chunk is { answer: string } {
  return typeof chunk === "object" && chunk !== null && "answer" in chunk;
}

function Main() {
  const { createMessage, appendToMessage } = useMessageContext();

  const messagesRef = useRef<HTMLDivElement>(null);

  async function handleSubmit(question: string) {
    if (question.trim() === "") {
      return;
    }

    createMessage("You", question);

    messagesRef.current?.scrollIntoView({ behavior: "smooth" });

    const id = createMessage("ChatDND");

    // const stream = await chain.stream({ question });
    const stream = await chain.stream(question);

    for await (const chunk of stream) {
      if (isStatusCode(chunk) && chunk.status_code !== 200) {
        appendToMessage(id, `An error occurred: ${chunk.message}`);
      }

      if (isAnswer(chunk)) {
        appendToMessage(id, chunk.answer);
      }
    }

    messagesRef.current?.scrollIntoView({ behavior: "smooth" });
  }

  return (
    <div className="relative z-0 flex flex-col h-screen w-full overflow-hidden">
      <header className="w-full p-2 border-b border-gray-300">
        <h1 className="text-center text-lg font-medium">ChatDND</h1>
      </header>
      <main className="relative h-full w-full flex-1 overflow-auto transition-width">
        <div role="presentation" className="flex h-full flex-col">
          <div className="flex-1 overflow-hidden">
            <div className="relative h-full">
              <div className="absolute left-0 right-0">
                <div className="h-1.5"></div>
              </div>
              <div
                className="flex h-full flex-col items-center overflow-y-scroll"
                ref={messagesRef}
              >
                <Messages />
              </div>
            </div>
          </div>
          <div className="w-full pt-2 md:pt-0 md:border-transparent">
            <PromptForm onSubmit={handleSubmit} />
            <div className="relative px-2 py-2 text-center text-xs text-gray-500 md:px-[60px]">
              <span>
                ChatDND can make mistakes. Consider checking important
                information.
              </span>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

function Messages() {
  const { messages } = useMessageContext();

  if (messages.length === 0) {
    return (
      <div className="my-5 text-2xl font-medium">How can I help you today?</div>
    );
  }

  return (
    <div className="w-full py-4">
      {messages.map((message) => (
        <div key={message.id} className="w-full">
          <div className="px-4 py-2 justify-center text-base md:gap-6 m-auto">
            <div className="relative flex w-full flex-col gap-2">
              <div className="font-semibold select-none">{message.author}</div>
              <div className="flex-col gap-1 md:gap-3">
                <div className="flex flex-grow flex-col max-w-full">
                  <div className="flex flex-col items-start gap-3 whitespace-pre-wrap break-words overflow-x-auto">
                    <div className="w-full break-words">
                      {message.text === "" ? (
                        <LoadingMessage />
                      ) : (
                        <p>{message.text.trim()}</p>
                      )}
                    </div>
                  </div>
                </div>
                <div className="pr-2 lg:pr-0"></div>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function LoadingMessage() {
  return (
    <div className="animate-pulse flex-1 space-y-3 py-1">
      <div className="h-2 bg-slate-200 rounded"></div>
      <div className="h-2 bg-slate-200 rounded"></div>
      <div className="h-2 bg-slate-200 rounded"></div>
    </div>
  );
}

interface FormElements extends HTMLFormControlsCollection {
  prompt: HTMLInputElement;
}

interface ChatFormElement extends HTMLFormElement {
  readonly elements: FormElements;
}

function PromptForm({ onSubmit }: { onSubmit: (query: string) => void }) {
  const [query, setQuery] = useState("");

  const disabled = query.trim().length === 0;

  async function handleSubmit(event: React.FormEvent<ChatFormElement>) {
    event.preventDefault();
    setQuery("");
    onSubmit(query);
  }

  return (
    <form
      className="stretch mx-2 flex flex-row gap-3 last:mb-2 md:mx-4 md:last:mb-6 lg:mx-auto lg:max-w-2xl xl:max-w-3xl"
      onSubmit={handleSubmit}
    >
      <div className="relative flex h-full flex-1 flex-col">
        <div className="flex w-full items-center">
          <div className="overflow-hidden flex flex-col w-full flex-grow relative border rounded-2xl">
            <textarea
              id="prompt"
              tabIndex={0}
              rows={1}
              placeholder="Message ChatDND..."
              className="m-0 w-full resize-none border-0 bg-transparent focus:ring-0 focus-visible:ring-0 py-[10px] pr-10 md:py-3.5 md:pr-12 max-h-48 placeholder-black/50 pl-3 md:pl-4"
              value={query}
              onChange={(event) => setQuery(event.currentTarget.value)}
            ></textarea>
            <button
              type="submit"
              className="absolute bottom-1.5 right-2 rounded-lg border border-black bg-black p-0.5 text-white transition-colors enabled:bg-black disabled:text-gray-400 disabled:opacity-10 md:bottom-3 md:right-3"
              disabled={disabled}
            >
              <span data-state="closed">
                <svg
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  className="text-white"
                >
                  <path
                    d="M7 11L12 6L17 11M12 18V7"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  ></path>
                </svg>
              </span>
            </button>
          </div>
        </div>
      </div>
    </form>
  );
}
