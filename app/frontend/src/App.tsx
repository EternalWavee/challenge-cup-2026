import {
  ArrowLeftRight,
  Check,
  ChevronDown,
  Crosshair,
  Download,
  Gauge,
  ImagePlus,
  Layers3,
  MapPin,
  Maximize2,
  Menu,
  MessageSquarePlus,
  PanelLeftClose,
  ScanLine,
  Send,
  Trash2,
  X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type {
  ChangeEvent,
  ClipboardEvent,
  DragEvent,
  KeyboardEvent as ReactKeyboardEvent,
} from "react";
import {
  deleteConversation,
  fileToDataUrl,
  loadConversations,
  saveConversation,
} from "./conversationStore";
import type { StoredConversation } from "./conversationStore";

type Health = {
  ok: boolean;
  status: "idle" | "loading" | "ready" | "error";
  error?: string | null;
  model: string;
  adapter: string;
  device: string;
};

type SelectedImage = {
  id: string;
  file: File;
  url: string;
};

type InferenceMode = "fast" | "quality" | "precision";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  imageUrls?: string[];
  meta?: string;
  inferenceMode?: InferenceMode;
  inferenceLabel?: string;
  grounding?: {
    visualizationUrl: string;
    boxes: GroundingBox[];
  };
  segmentation?: {
    visualizationUrl: string;
    targets: string[];
    areaRatios: Record<string, number>;
  };
};

type Conversation = StoredConversation<Message>;

type GroundingBox = {
  index: number;
  label: string;
  normalized: [number, number, number, number];
  pixels: [number, number, number, number];
};

type PreviewImage = {
  url: string;
  alt: string;
};

const EXAMPLES = [
  "图中一共有多少辆小汽车？",
  "请定位图中的灰色体育场。",
  "这是什么类型的场景？",
  "请描述这张遥感图像。",
  "请分割图中的道路区域。",
  "请描述两个时相之间的变化。",
];

const MAX_IMAGES = 2;
const MAX_IMAGE_BYTES = 15 * 1024 * 1024;
const INFERENCE_MODE_STORAGE_KEY = "tianshu-inference-mode";
const INFERENCE_MODES: Array<{
  id: InferenceMode;
  label: string;
  description: string;
}> = [
  {
    id: "fast",
    label: "快速",
    description: "响应优先",
  },
  {
    id: "quality",
    label: "增强",
    description: "细节与速度平衡",
  },
  {
    id: "precision",
    label: "高精定位",
    description: "单图精细定位",
  },
];

function storedInferenceMode(): InferenceMode {
  const value = window.localStorage.getItem(INFERENCE_MODE_STORAGE_KEY);
  return value === "fast" || value === "quality" || value === "precision"
    ? value
    : "quality";
}

function inferenceModeLabel(mode?: InferenceMode) {
  return INFERENCE_MODES.find((item) => item.id === mode)?.label ?? "增强";
}

function inferenceModeIcon(mode: InferenceMode) {
  if (mode === "fast") return Gauge;
  if (mode === "precision") return Crosshair;
  return ScanLine;
}

function makeId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function formatConversationTime(timestamp: number) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(timestamp);
}

function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(
    null,
  );
  const [historyReady, setHistoryReady] = useState(false);
  const [images, setImages] = useState<SelectedImage[]>([]);
  const [prompt, setPrompt] = useState("");
  const [inferenceMode, setInferenceMode] = useState<InferenceMode>(
    storedInferenceMode,
  );
  const [sending, setSending] = useState(false);
  const [notice, setNotice] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [modeMenuOpen, setModeMenuOpen] = useState(false);
  const [previewImage, setPreviewImage] = useState<PreviewImage | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const previewDialogRef = useRef<HTMLDialogElement>(null);
  const conversationsRef = useRef<Conversation[]>([]);
  const isComposingRef = useRef(false);
  const modeMenuRef = useRef<HTMLDivElement>(null);

  const messages =
    conversations.find((conversation) => conversation.id === activeConversationId)
      ?.messages ?? [];

  useEffect(() => {
    let active = true;

    const poll = async () => {
      try {
        const response = await fetch("/api/health");
        if (!response.ok) throw new Error("服务状态获取失败");
        const data = (await response.json()) as Health;
        if (active) setHealth(data);
      } catch {
        if (active) {
          setHealth({
            ok: false,
            status: "error",
            error: "后端服务未连接",
            model: "",
            adapter: "",
            device: "",
          });
        }
      }
    };

    void poll();
    const timer = window.setInterval(poll, 3000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    window.localStorage.setItem(INFERENCE_MODE_STORAGE_KEY, inferenceMode);
  }, [inferenceMode]);

  useEffect(() => {
    if (!modeMenuOpen) return;

    const closeOnOutsideClick = (event: MouseEvent) => {
      if (!modeMenuRef.current?.contains(event.target as Node)) {
        setModeMenuOpen(false);
      }
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setModeMenuOpen(false);
    };

    document.addEventListener("mousedown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("mousedown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [modeMenuOpen]);

  useEffect(() => {
    let active = true;

    void loadConversations<Message>()
      .then((records) => {
        if (!active) return;
        conversationsRef.current = records;
        setConversations(records);
        setActiveConversationId(null);
      })
      .catch(() => {
        if (active) setNotice("历史记录暂时不可用，本次对话仍可正常使用。");
      })
      .finally(() => {
        if (active) setHistoryReady(true);
      });

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

  useEffect(() => {
    const dialog = previewDialogRef.current;
    if (!dialog) return;

    if (previewImage && !dialog.open) dialog.showModal();
    if (!previewImage && dialog.open) dialog.close();
  }, [previewImage]);

  const appendFiles = (files: File[]) => {
    setNotice("");
    const available = MAX_IMAGES - images.length;
    if (available <= 0) {
      setNotice("每次最多上传两张图像。");
      return;
    }

    const valid = files
      .filter((file) => {
        if (!file.type.startsWith("image/")) {
          setNotice("只支持常见图像文件。");
          return false;
        }
        if (file.size > MAX_IMAGE_BYTES) {
          setNotice(`图像 ${file.name} 超过 15 MB。`);
          return false;
        }
        return true;
      })
      .slice(0, available)
      .map((file) => ({ id: makeId(), file, url: URL.createObjectURL(file) }));

    setImages((current) => [...current, ...valid]);
  };

  const onFiles = (event: ChangeEvent<HTMLInputElement>) => {
    appendFiles(Array.from(event.target.files ?? []));
    event.target.value = "";
  };

  const onPaste = (event: ClipboardEvent<HTMLTextAreaElement>) => {
    const files = Array.from(event.clipboardData.items)
      .filter((item) => item.kind === "file" && item.type.startsWith("image/"))
      .map((item) => item.getAsFile())
      .filter((file): file is File => file !== null);
    if (files.length > 0) {
      event.preventDefault();
      appendFiles(files);
    }
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    appendFiles(Array.from(event.dataTransfer.files));
  };

  const removeImage = (id: string) => {
    setImages((current) => {
      const target = current.find((image) => image.id === id);
      if (target) URL.revokeObjectURL(target.url);
      return current.filter((image) => image.id !== id);
    });
  };

  const swapImages = () => setImages((current) => [...current].reverse());

  const clearComposer = () => {
    images.forEach((image) => URL.revokeObjectURL(image.url));
    setImages([]);
    setPrompt("");
    setNotice("");
    setPreviewImage(null);
  };

  const upsertConversation = (conversation: Conversation) => {
    const next = [
      conversation,
      ...conversationsRef.current.filter((item) => item.id !== conversation.id),
    ].sort((left, right) => right.updatedAt - left.updatedAt);
    conversationsRef.current = next;
    setConversations(next);
    void saveConversation(conversation).catch(() =>
      setNotice("会话已显示，但未能写入浏览器历史。"),
    );
  };

  const appendConversationMessage = (conversationId: string, message: Message) => {
    const conversation = conversationsRef.current.find(
      (item) => item.id === conversationId,
    );
    if (!conversation) return;
    upsertConversation({
      ...conversation,
      updatedAt: Date.now(),
      messages: [...conversation.messages, message],
    });
  };

  const startNewConversation = () => {
    clearComposer();
    setActiveConversationId(null);
    setSidebarOpen(false);
  };

  const selectConversation = (conversationId: string) => {
    clearComposer();
    setActiveConversationId(conversationId);
    setSidebarOpen(false);
  };

  const removeConversation = (conversationId: string) => {
    const next = conversationsRef.current.filter(
      (conversation) => conversation.id !== conversationId,
    );
    conversationsRef.current = next;
    setConversations(next);
    void deleteConversation(conversationId).catch(() =>
      setNotice("未能从浏览器中删除这条历史记录。"),
    );
    if (activeConversationId === conversationId) {
      clearComposer();
      setActiveConversationId(null);
    }
  };

  const sendMessage = async () => {
    const cleanPrompt = prompt.trim();
    if (!cleanPrompt || sending) return;
    if (!historyReady) {
      setNotice("会话记录仍在载入，请稍候。");
      return;
    }
    if (images.length === 0) {
      setNotice("请先上传或粘贴至少一张遥感图像。");
      return;
    }

    const outgoing = [...images];
    const conversationId = activeConversationId ?? makeId();
    const started = performance.now();
    setImages([]);
    setPrompt("");
    setNotice("");
    setSending(true);

    const form = new FormData();
    form.append("prompt", cleanPrompt);
    form.append("inference_mode", inferenceMode);
    outgoing.forEach((image) => form.append("images", image.file, image.file.name));

    try {
      const persistentImageUrls = await Promise.all(
        outgoing.map((image) => fileToDataUrl(image.file)),
      );
      outgoing.forEach((image) => URL.revokeObjectURL(image.url));
      const userMessage: Message = {
        id: makeId(),
        role: "user",
        content: cleanPrompt,
        imageUrls: persistentImageUrls,
        inferenceMode,
      };

      if (activeConversationId) {
        appendConversationMessage(conversationId, userMessage);
      } else {
        const conversation: Conversation = {
          id: conversationId,
          title: cleanPrompt.slice(0, 32),
          updatedAt: Date.now(),
          messages: [userMessage],
        };
        upsertConversation(conversation);
        setActiveConversationId(conversationId);
      }

      const response = await fetch("/api/chat", { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) {
        const detail =
          typeof data.detail === "string"
            ? data.detail
            : data.detail?.message ?? "推理请求失败";
        throw new Error(detail);
      }
      appendConversationMessage(conversationId, {
          id: makeId(),
          role: "assistant",
          content: data.grounding
            ? `已定位 ${data.grounding.boxes.length} 个目标。`
            : data.answer,
          meta: `推理 ${(Number(data.elapsed_ms) / 1000).toFixed(1)} 秒`,
          inferenceMode: data.inference?.requested_mode ?? inferenceMode,
          inferenceLabel: data.inference?.label ?? inferenceModeLabel(inferenceMode),
          grounding: data.grounding
            ? {
                visualizationUrl: data.grounding.visualization_url,
                boxes: data.grounding.boxes,
              }
            : undefined,
          segmentation: data.segmentation?.visualization_url
            ? {
                visualizationUrl: data.segmentation.visualization_url,
                targets: data.segmentation.targets ?? [],
                areaRatios: data.segmentation.area_ratios ?? {},
              }
            : undefined,
        });
    } catch (error) {
      outgoing.forEach((image) => URL.revokeObjectURL(image.url));
      appendConversationMessage(conversationId, {
          id: makeId(),
          role: "assistant",
          content: `请求失败：${error instanceof Error ? error.message : "未知错误"}`,
          meta: `等待 ${((performance.now() - started) / 1000).toFixed(1)} 秒`,
        });
    } finally {
      setSending(false);
    }
  };

  const onKeyDown = (event: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if (
      isComposingRef.current ||
      event.nativeEvent.isComposing ||
      event.nativeEvent.keyCode === 229
    ) {
      return;
    }
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (sending) {
        return;
      }
      void sendMessage();
    }
  };

  const statusLabel =
    health?.status === "ready"
      ? "天枢就绪"
      : health?.status === "loading"
        ? "天枢加载中"
        : health?.status === "error"
          ? "天枢未连接"
          : "等待天枢";
  const ActiveModeIcon = inferenceModeIcon(inferenceMode);

  return (
    <div className="app-shell">
      <button
        className="mobile-menu"
        type="button"
        aria-label="打开侧栏"
        onClick={() => setSidebarOpen(true)}
      >
        <Menu size={20} />
      </button>

      {sidebarOpen && (
        <button
          className="sidebar-scrim"
          type="button"
          aria-label="关闭侧栏"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside className={`sidebar ${sidebarOpen ? "is-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <img src="/sjtu-emblem.png" alt="" />
          </div>
          <div className="brand-copy">
            <span className="brand-school">上海交通大学</span>
            <strong>天枢·遥感</strong>
          </div>
          <button
            className="sidebar-close"
            type="button"
            aria-label="关闭侧栏"
            onClick={() => setSidebarOpen(false)}
          >
            <PanelLeftClose size={19} />
          </button>
        </div>

        <button
          className="new-chat"
          type="button"
          disabled={sending}
          onClick={startNewConversation}
        >
          <MessageSquarePlus size={18} />
          新建对话
        </button>

        <nav className="history" aria-label="历史对话">
          <div className="history-heading">
            <span>历史记录</span>
            <span>{conversations.length}</span>
          </div>
          {!historyReady ? (
            <p className="history-empty">正在载入…</p>
          ) : conversations.length === 0 ? (
            <p className="history-empty">完成一次解译后会保存在这里</p>
          ) : (
            <div className="history-list">
              {conversations.map((conversation) => (
                <div
                  className={`history-item ${
                    conversation.id === activeConversationId ? "is-active" : ""
                  }`}
                  key={conversation.id}
                >
                  <button
                    className="history-select"
                    type="button"
                    disabled={sending}
                    onClick={() => selectConversation(conversation.id)}
                  >
                    <strong>{conversation.title}</strong>
                    <span>{formatConversationTime(conversation.updatedAt)}</span>
                  </button>
                  <button
                    className="history-delete"
                    type="button"
                    disabled={sending}
                    aria-label={`删除对话：${conversation.title}`}
                    onClick={() => removeConversation(conversation.id)}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </nav>

        <section className="model-card" aria-label="模型状态">
          <div className="status-row">
            <span className={`status-dot status-${health?.status ?? "idle"}`} />
            <strong>{statusLabel}</strong>
          </div>
        </section>
      </aside>

      <main className="workspace">
        <section className="conversation" aria-live="polite">
          {messages.length === 0 ? (
            <div className="empty-state">
              <img
                className="empty-watermark"
                src="/sjtu-emblem.png"
                alt=""
                aria-hidden="true"
              />
              <span className="empty-kicker">天枢·遥感</span>
              <h1>遥感影像智能解译</h1>
              <p>
                粘贴、拖入或选择一至两张图像，直接描述你想知道的内容。
              </p>
              <div className="examples">
                {EXAMPLES.map((example, index) => (
                  <button
                    key={example}
                    type="button"
                    onClick={() => setPrompt(example)}
                  >
                    <span>{String(index + 1).padStart(2, "0")}</span>
                    <strong>{example}</strong>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message) => (
                <article
                  className={`message message-${message.role}`}
                  key={message.id}
                >
                  {message.role === "assistant" && (
                    <div className="message-avatar" aria-hidden="true">
                      <img src="/sjtu-emblem.png" alt="" />
                    </div>
                  )}
                  <div className="message-body">
                    {message.role === "user" ? (
                      <div
                        className={`user-message ${
                        message.imageUrls?.length === 2 ? "has-image-pair" : ""
                        }`}
                      >
                        {message.imageUrls && (
                          <div className="message-images">
                            {message.imageUrls.map((url, index) => (
                              <figure key={url}>
                                <img src={url} alt={`输入影像 ${index + 1}`} />
                                {message.imageUrls!.length === 2 && (
                                  <figcaption>时相{index === 0 ? "一" : "二"}</figcaption>
                                )}
                              </figure>
                            ))}
                          </div>
                        )}
                        <div className="user-bubble">
                          <div className="message-content">{message.content}</div>
                        </div>
                        {message.inferenceMode && (
                          <div className="message-mode">
                            {inferenceModeLabel(message.inferenceMode)}模式
                          </div>
                        )}
                      </div>
                    ) : (
                      <>
                        <div className="message-label">天枢</div>
                        <div className="assistant-response">
                      <div className="message-content">{message.content}</div>
                      {message.grounding && (
                      <section className="grounding-result" aria-label="定位可视化结果">
                        <div className="grounding-heading">
                          <MapPin size={17} />
                          <strong>定位结果</strong>
                          <span>归一化坐标 0-1000</span>
                          <div className="grounding-actions">
                            <button
                              type="button"
                              onClick={() =>
                                setPreviewImage({
                                  url: message.grounding!.visualizationUrl,
                                  alt: "带有目标定位框的遥感影像",
                                })
                              }
                            >
                              <Maximize2 size={15} />
                              查看大图
                            </button>
                            <a
                              href={message.grounding.visualizationUrl}
                              download="天枢-定位结果.jpg"
                            >
                              <Download size={15} />
                              下载结果
                            </a>
                          </div>
                        </div>
                        <div className="visual-result-content">
                          <button
                            className="grounding-image-button"
                            type="button"
                            aria-label="查看定位结果大图"
                            onClick={() =>
                              setPreviewImage({
                                url: message.grounding!.visualizationUrl,
                                alt: "带有目标定位框的遥感影像",
                              })
                            }
                          >
                            <img
                              src={message.grounding.visualizationUrl}
                              alt="带有目标定位框的遥感影像"
                            />
                          </button>
                          <ol>
                            {message.grounding.boxes.map((box) => (
                              <li key={`${box.index}-${box.normalized.join("-")}`}>
                                <span>
                                  {box.index}. {box.label}
                                </span>
                                <code>[{box.normalized.join(", ")}]</code>
                              </li>
                            ))}
                          </ol>
                        </div>
                      </section>
                      )}
                      {message.segmentation && (
                      <section className="grounding-result" aria-label="语义分割可视化结果">
                        <div className="grounding-heading">
                          <Layers3 size={17} />
                          <strong>语义分割结果</strong>
                          <span>像素级掩膜</span>
                          <div className="grounding-actions">
                            <button
                              type="button"
                              onClick={() =>
                                setPreviewImage({
                                  url: message.segmentation!.visualizationUrl,
                                  alt: "遥感影像语义分割叠加结果",
                                })
                              }
                            >
                              <Maximize2 size={15} />
                              查看大图
                            </button>
                            <a
                              href={message.segmentation.visualizationUrl}
                              download="天枢-语义分割结果.png"
                            >
                              <Download size={15} />
                              下载结果
                            </a>
                          </div>
                        </div>
                        <div className="visual-result-content is-single">
                          <button
                            className="grounding-image-button"
                            type="button"
                            aria-label="查看语义分割结果大图"
                            onClick={() =>
                              setPreviewImage({
                                url: message.segmentation!.visualizationUrl,
                                alt: "遥感影像语义分割叠加结果",
                              })
                            }
                          >
                            <img
                              src={message.segmentation.visualizationUrl}
                              alt="遥感影像语义分割叠加结果"
                            />
                          </button>
                        </div>
                      </section>
                      )}
                      {message.meta && (
                        <div className="message-meta">
                          {message.inferenceLabel
                            ? `${message.inferenceLabel} | ${message.meta}`
                            : message.meta}
                        </div>
                      )}
                        </div>
                      </>
                    )}
                  </div>
                </article>
              ))}
              {sending && (
                <article className="message message-assistant">
                  <div className="message-avatar">
                    <img src="/sjtu-emblem.png" alt="" />
                  </div>
                  <div className="message-body">
                    <div className="message-label">天枢</div>
                    <div className="assistant-response assistant-response-thinking">
                      <div className="thinking" aria-live="polite">
                        <span>正在思考</span>
                        <span className="thinking-ellipsis" aria-hidden="true">…</span>
                      </div>
                    </div>
                  </div>
                </article>
              )}
            </div>
          )}
          <div ref={bottomRef} />
        </section>

        <section className="composer-wrap">
          <div
            className="composer"
            onDragOver={(event) => event.preventDefault()}
            onDrop={onDrop}
          >
            {images.length > 0 && (
              <div className="selected-images">
                {images.map((image, index) => (
                  <figure key={image.id}>
                    <img src={image.url} alt={`待发送影像 ${index + 1}`} />
                    <figcaption>
                      {images.length === 2 ? `时相${index === 0 ? "一" : "二"}` : image.file.name}
                    </figcaption>
                    <button
                      type="button"
                      aria-label={`移除 ${image.file.name}`}
                      onClick={() => removeImage(image.id)}
                    >
                      <X size={15} />
                    </button>
                  </figure>
                ))}
                {images.length === 2 && (
                  <button
                    className="swap-button"
                    type="button"
                    onClick={swapImages}
                    title="交换时相顺序"
                  >
                    <ArrowLeftRight size={17} />
                    交换顺序
                  </button>
                )}
              </div>
            )}

            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              onKeyDown={onKeyDown}
              onCompositionStart={() => {
                isComposingRef.current = true;
              }}
              onCompositionEnd={() => {
                window.setTimeout(() => {
                  isComposingRef.current = false;
                }, 0);
              }}
              onPaste={onPaste}
              placeholder="询问这张遥感影像……"
              rows={1}
              maxLength={4000}
              aria-label="输入问题"
            />

            <div className="composer-actions">
              <div className="upload-actions">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={sending || images.length >= MAX_IMAGES}
                >
                  <ImagePlus size={18} />
                  添加图像
                </button>
                <span>支持粘贴或拖入，最多 2 张</span>
              </div>
              <div className="composer-controls">
                <div className="mode-switcher" ref={modeMenuRef}>
                  <button
                    className="mode-trigger"
                    type="button"
                    aria-haspopup="menu"
                    aria-expanded={modeMenuOpen}
                    aria-controls="inference-mode-menu"
                    onClick={() => setModeMenuOpen((open) => !open)}
                  >
                    <ActiveModeIcon size={15} aria-hidden="true" />
                    <strong>{inferenceModeLabel(inferenceMode)}</strong>
                    <ChevronDown
                      className={modeMenuOpen ? "is-open" : ""}
                      size={14}
                      aria-hidden="true"
                    />
                  </button>

                  {modeMenuOpen && (
                    <div
                      className="mode-panel"
                      id="inference-mode-menu"
                      role="menu"
                      aria-label="选择推理模式"
                    >
                      <div className="mode-panel-heading">
                        <strong>推理模式</strong>
                        <span>按任务选择</span>
                      </div>
                      <div className="mode-options">
                        {INFERENCE_MODES.map((mode) => {
                          const selected = inferenceMode === mode.id;
                          const ModeIcon = inferenceModeIcon(mode.id);
                          return (
                            <button
                              key={mode.id}
                              type="button"
                              role="menuitemradio"
                              aria-checked={selected}
                              className={selected ? "is-selected" : ""}
                              disabled={sending}
                              onClick={() => {
                                setInferenceMode(mode.id);
                                setModeMenuOpen(false);
                              }}
                            >
                              <ModeIcon
                                className="mode-option-icon"
                                size={17}
                                aria-hidden="true"
                              />
                              <span>
                                <strong>{mode.label}</strong>
                                <small>{mode.description}</small>
                              </span>
                              {selected && <Check size={16} aria-hidden="true" />}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
                <button
                  className="send-button"
                  type="button"
                  aria-label="发送"
                  disabled={
                    sending ||
                    !historyReady ||
                    !prompt.trim() ||
                    images.length === 0
                  }
                  onClick={() => void sendMessage()}
                >
                  <Send size={18} />
                </button>
              </div>
            </div>

            <input
              ref={fileInputRef}
              className="visually-hidden"
              type="file"
              accept="image/*"
              multiple
              onChange={onFiles}
            />
          </div>
          <div className="composer-foot">
            <span>
              {sending
                ? "推理中，可继续输入，完成后即可发送"
                : notice || "Enter 发送 · Shift + Enter 换行"}
            </span>
            {activeConversationId && (
              <button
                type="button"
                disabled={sending}
                onClick={() => removeConversation(activeConversationId)}
              >
                <Trash2 size={14} />
                删除对话
              </button>
            )}
          </div>
        </section>
      </main>

      <dialog
        ref={previewDialogRef}
        className="image-lightbox"
        aria-label="定位结果大图"
        onClose={() => setPreviewImage(null)}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            event.currentTarget.close();
          }
        }}
        onClick={(event) => {
          if (event.target === event.currentTarget) event.currentTarget.close();
        }}
      >
        {previewImage && (
          <div className="image-lightbox-panel">
            <div className="image-lightbox-toolbar">
              <strong>定位结果</strong>
              <div>
                <a href={previewImage.url} download="天枢-定位结果.jpg">
                  <Download size={16} />
                  下载
                </a>
                <button
                  type="button"
                  aria-label="关闭大图"
                  onClick={() => previewDialogRef.current?.close()}
                >
                  <X size={18} />
                  关闭
                </button>
              </div>
            </div>
            <img
              className="image-lightbox-image"
              src={previewImage.url}
              alt={previewImage.alt}
            />
          </div>
        )}
      </dialog>
    </div>
  );
}

export default App;
