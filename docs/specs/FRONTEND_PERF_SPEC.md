# Technical Spec: Frontend Performance & Rendering Optimization

## Goal
Ensure the Research Dashboard maintains a Lighthouse Performance score of 90+ even with complex knowledge graphs.

---

## 1. Rendering Optimization
*   **Virtualization**: Use `react-window` or `react-virtualized` for the search results list if results > 100.
*   **Memoization**: Use `React.memo` for static card components (e.g., `StatusChip`) to prevent re-renders on global state changes.
*   **Debouncing**: The `SearchBar` MUST debounce inputs by 300ms before triggering an RTK Query.

---

## 2. Dynamic Code Splitting
*   **Lazy Loading**: Heavy visualization libraries (D3.js, Recharts) must be loaded using `next/dynamic`.
```typescript
const KnowledgeGraph = dynamic( => import('./KnowledgeGraph'), {
  ssr: false,
  loading:  => <Skeleton variant="rectangular" height={400} />,
});
```

---

## 3. Data Fetching (RTK Query)
*   **Cache Strategy**: Cache search results for 5 minutes.
*   **Polling**: The `IngestionMonitor` will poll every 5s instead of using WebSockets for "Mobile" mode to save battery.

---

## 4. Verification Metrics
*   **First Contentful Paint (FCP)**: < 1.2s.
*   **Time to Interactive (TTI)**: < 2.5s.
*   **Cumulative Layout Shift (CLS)**: < 0.1.
