# Technical Spec: IngestForge Web Portal Architecture

## Goal
Deliver a world-class research dashboard using Next.js, Material UI, and Redux, optimized for performance (Lighthouse > 90) and scalability.

---

## 1. Technology Stack
*   **Framework**: Next.js 14+ (App Router) - For SSR and SEO optimization.
*   **Language**: TypeScript - 100% type safety for component props and state.
*   **Styling**: Material UI (MUI) v5+ - For professional, responsive design.
*   **State Management**: Redux Toolkit (RTK) - For global state (Auth, Project Settings).
*   **API Layer**: RTK Query - For caching and real-time backend synchronization.
*   **Visualization**: Recharts / Nivo - For concept maps and study dashboards.

---

## 2. Directory Structure (`frontend/`)
```text
frontend/
├── src/
│   ├── app/            # Next.js App Router (Pages & Layouts)
│   ├── components/     # Atomic UI Components (Buttons, Cards)
│   ├── features/       # Logical Slices (auth, research, study)
│   │   ├── auth/
│   │   │   ├── authSlice.ts
│   │   │   └── AuthProvider.tsx
│   ├── store/          # Redux Store Configuration
│   ├── theme/          # MUI Theme Definition (Dark Mode)
│   └── lib/            # Utilities & API Clients
```

---

## 3. Theming & Design System
*   **Primary Palette**: `#1a1a2e` (Navy), `#e94560` (Crimson).
*   **Typography**: Inter or Roboto (Standard Material font).
*   **Components**: Use MUI `Responsive Container` and `Grid` for cross-browser/device compatibility.

---

## 4. Performance & QA (Lighthouse 90+)
*   **Optimization**:
    *   Lazy load heavy visualization components.
    *   Use `next/image` for any assets.
    *   Implement `MUI bundle splitting` to reduce first-paint time.
*   **Accessibility**:
    *   100% ARIA label coverage.
    *   Keyboard navigation support for all dashboard elements.
    *   Contrast ratio > 4.5:1.

---

## 5. Implementation Strategy
1.  **Scaffold**: `npx create-next-app@latest frontend --typescript --tailwind --eslint`.
2.  **MUI Setup**: Install `@mui/material @emotion/react @emotion/styled`.
3.  **Redux Integration**: Install `@reduxjs/toolkit react-redux`.
4.  **Mock API**: Start with RTK Query pointing to a local FastAPI dev server.
