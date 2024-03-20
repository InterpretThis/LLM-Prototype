import { RemoteRunnable } from "@langchain/core/runnables/remote";

export const chain = new RemoteRunnable({
  url: "http://127.0.0.1:5001/api",
});
