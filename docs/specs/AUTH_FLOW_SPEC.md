# Technical Spec: User Authentication Flow

## Goal
Implement a secure, JWT-based authentication handshake between the React Portal and the FastAPI Backend.

---

## 1. Handshake Protocol
1.  **Request**: User submits `credentials` to `POST /v1/auth/login`.
2.  **Validate**: Backend asserts user existence and verifies password hash.
3.  **Response**: Returns `access_token` (JWT) and `refresh_token`.
4.  **Store**: Frontend Redux `authSlice` captures token; persists to `localStorage`.
5.  **Authorize**: All subsequent RTK Query calls include `Authorization: Bearer <token>` header.

---

## 2. Token Management (Frontend)
**Module**: `frontend/src/features/auth/authMiddleware.ts`
*   **Auto-Logout**: If a request returns `401 Unauthorized`, trigger `clearAuth` action.
*   **Persistence**: On app boot, check `localStorage`. If token valid, `setCredentials`.

---

## 3. UI Requirements (Rule #7: Input Validation)
*   **Validation**: Use `yup` or `zod` for client-side form validation.
*   **Feedback**: Display specific error messages (e.g., "Invalid email format") rather than generic "Login failed."
*   **Security**: Password fields must use `type="password"`.

---

## 4. Acceptance Criteria
- [ ] User can log in and see their profile name in the SideBar.
- [ ] Direct navigation to `/dashboard` without a token redirects to `/login`.
- [ ] "Log Out" button clears all state and local storage.
