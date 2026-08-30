const DATABASE_NAME = "tianshu-remote-sensing";
const DATABASE_VERSION = 1;
const STORE_NAME = "conversations";

export type StoredConversation<TMessage> = {
  id: string;
  title: string;
  updatedAt: number;
  messages: TMessage[];
};

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);

    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("无法打开会话数据库"));
  });
}

function waitForTransaction(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () =>
      reject(transaction.error ?? new Error("无法保存会话记录"));
    transaction.onabort = () =>
      reject(transaction.error ?? new Error("会话记录写入已中止"));
  });
}

export async function loadConversations<TMessage>(): Promise<
  StoredConversation<TMessage>[]
> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(STORE_NAME, "readonly");
    const request = transaction.objectStore(STORE_NAME).getAll();
    const records = await new Promise<StoredConversation<TMessage>[]>(
      (resolve, reject) => {
        request.onsuccess = () =>
          resolve(request.result as StoredConversation<TMessage>[]);
        request.onerror = () =>
          reject(request.error ?? new Error("无法读取会话记录"));
      },
    );
    return records.sort((left, right) => right.updatedAt - left.updatedAt);
  } finally {
    database.close();
  }
}

export async function saveConversation<TMessage>(
  conversation: StoredConversation<TMessage>,
): Promise<void> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(STORE_NAME, "readwrite");
    transaction.objectStore(STORE_NAME).put(conversation);
    await waitForTransaction(transaction);
  } finally {
    database.close();
  }
}

export async function deleteConversation(id: string): Promise<void> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(STORE_NAME, "readwrite");
    transaction.objectStore(STORE_NAME).delete(id);
    await waitForTransaction(transaction);
  } finally {
    database.close();
  }
}

export function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error ?? new Error("无法保存输入图像"));
    reader.readAsDataURL(file);
  });
}
