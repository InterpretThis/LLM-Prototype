import { v4 as uuid } from "uuid";
import React, { createContext, useContext, useState } from "react";

export type Author = "You" | "ChatDND";

export interface Message {
  id: string;
  author: Author;
  text: string;
  timestamp: Date;
}

export interface MessageContext {
  messages: Message[];
  createMessage: (author: Author, text?: string) => string;
  appendToMessage: (id: string, text: string) => void;
}

const MessageContext = createContext<MessageContext | null>(null);

export function useMessageContext() {
  const context = useContext(MessageContext);
  if (!context) {
    throw new Error("useMessageContext must be used within a MessageProvider");
  }
  return context;
}

export function MessageProvider({ children }: { children: React.ReactNode }) {
  const [messages, setMessages] = useState<Message[]>([]);

  const createMessage = (author: Author, text = "") => {
    const id = uuid();
    setMessages((messages) => [
      ...messages,
      { id, author, text, timestamp: new Date() },
    ]);
    return id;
  };

  const appendToMessage = (id: string, text: string) => {
    setMessages((messages) =>
      messages.map((message) =>
        message.id === id ? { ...message, text: message.text + text } : message
      )
    );
  };

  return (
    <MessageContext.Provider
      value={{ messages, createMessage, appendToMessage }}
    >
      {children}
    </MessageContext.Provider>
  );
}
