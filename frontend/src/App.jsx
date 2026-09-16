import { useEffect, useMemo, useRef, useState } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

// ---------- Small presentational components ----------

function BrandHeader({ uploading, onUpload }) {
  return (
    <header className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5 lg:px-10">
      <div className="flex items-center gap-3">
        <div className="brand-mark">A</div>
        <div>
          <p className="font-display text-xl font-bold tracking-tight">
            AnyCart
          </p>
          <p className="text-[11px] uppercase tracking-[0.24em] text-[#8b8275]">
            intelligent E-commerce Agent
          </p>
        </div>
      </div>

      <label className="upload-button">
        {uploading ? "Indexing..." : "Upload catalog PDF"}
        <input
          type="file"
          accept="application/pdf"
          onChange={onUpload}
          disabled={uploading}
        />
      </label>
    </header>
  );
}

function Hero() {
  return (
    <section className="hero-panel">
      <div>
        <p className="eyebrow">AI SHOPPING WORKSPACE</p>
        <h1 className="font-display mt-3 max-w-2xl text-5xl font-bold leading-[0.95] tracking-[-0.04em] text-white md:text-7xl">
          A smarter way to find what fits.
        </h1>
        <p className="mt-6 max-w-xl text-base leading-7 text-[#d8d2c7]">
          Explore your catalog through a conversational agent powered by
          LangGraph, LangChain RAG, and semantic search.
        </p>
      </div>
      <div className="hero-orbit">
        <span>RAG</span>
        <span>GRAPH</span>
        <span>SQL</span>
      </div>
    </section>
  );
}

function CategoryPills({ categories, active, onSelect }) {
  return (
    <div className="mb-6 flex flex-wrap gap-2">
      {categories.map((item) => (
        <button
          key={item}
          onClick={() => onSelect(item)}
          className={`category-pill ${active === item ? "active" : ""}`}
        >
          {item}
        </button>
      ))}
    </div>
  );
}

function ProductCard({ product }) {
  return (
    <article className="product-card">
      <div className="flex items-start justify-between">
        <span className="product-id">{product.product_id}</span>
        <span className="text-xs text-[#9b9388]">{product.category}</span>
      </div>
      <h3 className="font-display mt-8 text-xl font-bold">{product.name}</h3>
      <p className="mt-2 text-sm leading-6 text-[#77736c]">
        {product.description}
      </p>
      <div className="mt-6 text-2xl font-bold text-[#b45f35]">
        ${Number(product.price).toFixed(2)}
      </div>
    </article>
  );
}

function ProductGrid({ products }) {
  if (!products.length) {
    return (
      <div className="empty-card">
        Upload a catalog PDF to see featured products.
      </div>
    );
  }

  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
      {products.slice(0, 6).map((product) => (
        <ProductCard key={product.product_id} product={product} />
      ))}
    </div>
  );
}

function CatalogSection({
  catalogName,
  productCount,
  categories,
  category,
  onSelectCategory,
  notice,
  filtered,
}) {
  return (
    <section>
      <div className="mb-5 flex items-end justify-between">
        <div>
          <p className="eyebrow text-[#aa6a3d]">{catalogName || "CATALOG"}</p>
          <h2 className="font-display mt-2 text-3xl font-bold">
            Featured picks
          </h2>
        </div>
        <span className="rounded-full bg-white px-4 py-2 text-xs font-semibold text-[#8b8275]">
          {productCount} products
        </span>
      </div>

      <CategoryPills
        categories={categories}
        active={category}
        onSelect={onSelectCategory}
      />

      {notice && (
        <div className="mb-5 rounded-2xl border border-[#e7d9c4] bg-[#fffaf1] px-4 py-3 text-sm text-[#806648]">
          {notice}
        </div>
      )}

      <ProductGrid products={filtered} />
    </section>
  );
}

function ChatMessage({ message }) {
  return (
    <div className={`message ${message.role}`}>
      {message.role === "assistant" && <span className="avatar">A</span>}
      <div className={message.pending ? "pending" : ""}>{message.content}</div>
    </div>
  );
}

function ChatPanel({ messages, prompt, onPromptChange, onSubmit, chatEndRef }) {
  return (
    <section className="chat-panel max-h-135 overflow-hidden">
      <div className="flex items-center justify-between border-b border-white/10 pb-4">
        <div>
          <p className="eyebrow text-[#d59a73]">ASSISTANT</p>
          <h2 className="font-display mt-1 text-2xl font-bold text-white">
            Your shopping guide
          </h2>
        </div>
        <span className="status-dot" />
      </div>

      <div className="chat-history">
        {messages.map((message, index) => (
          <ChatMessage key={index} message={message} />
        ))}
        <div ref={chatEndRef} />
      </div>

      <form onSubmit={onSubmit} className="chat-form">
        <input
          value={prompt}
          onChange={(event) => onPromptChange(event.target.value)}
          placeholder="Ask about your catalog..."
        />
        <button type="submit">Send</button>
      </form>
    </section>
  );
}

// ---------- Main component ----------

function App() {
  const [products, setProducts] = useState([]);
  const [category, setCategory] = useState("All");
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Welcome to AnyCart. Upload a catalog PDF, then ask me about products or orders.",
    },
  ]);
  const [prompt, setPrompt] = useState("");
  const [uploading, setUploading] = useState(false);
  const [catalogName, setCatalogName] = useState("");
  const [notice, setNotice] = useState("");
  const chatEndRef = useRef(null);

  const categories = useMemo(
    () => ["All", ...new Set(products.map((p) => p.category))],
    [products],
  );

  const filteredProducts = useMemo(
    () =>
      category === "All"
        ? products
        : products.filter((p) => p.category === category),
    [products, category],
  );

  useEffect(() => {
    loadProducts();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function loadProducts() {
    const response = await fetch(`${API}/catalog/products`);
    if (response.ok) setProducts(await response.json());
  }

  async function uploadCatalog(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setNotice("Indexing catalog with LangChain and pgvector...");

    const form = new FormData();
    form.append("file", file);

    const response = await fetch(`${API}/catalog/upload`, {
      method: "POST",
      body: form,
    });
    const body = await response.json();

    if (!response.ok) {
      setNotice(body.detail || "Catalog upload failed.");
    } else {
      setCatalogName(body.filename);
      setNotice(`${body.products} products indexed from ${body.filename}`);
      await loadProducts();
    }

    setUploading(false);
  }

  async function sendMessage(event) {
    event?.preventDefault();
    const text = prompt.trim();
    if (!text) return;

    setPrompt("");
    setMessages((current) => [
      ...current,
      { role: "user", content: text },
      { role: "assistant", content: "Thinking...", pending: true },
    ]);

    const response = await fetch(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const body = await response.json();

    setMessages((current) => [
      ...current.slice(0, -1),
      {
        role: "assistant",
        content: response.ok
          ? body.response
          : body.detail || "The assistant could not respond.",
      },
    ]);
  }

  return (
    <div className="min-h-screen bg-[#f6f4ef] text-[#17212b]">
      <BrandHeader uploading={uploading} onUpload={uploadCatalog} />

      <main className="mx-auto max-w-7xl px-6 pb-12 lg:px-10">
        <Hero />

        <div className="mt-7 grid gap-7 lg:grid-cols-[1.6fr_0.9fr]">
          <CatalogSection
            catalogName={catalogName}
            productCount={products.length}
            categories={categories}
            category={category}
            onSelectCategory={setCategory}
            notice={notice}
            filtered={filteredProducts}
          />

          <ChatPanel
            messages={messages}
            prompt={prompt}
            onPromptChange={setPrompt}
            onSubmit={sendMessage}
            chatEndRef={chatEndRef}
          />
        </div>
      </main>
    </div>
  );
}

export default App;
