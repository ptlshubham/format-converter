# Agent Work Rules (Gemini Agent & AI Coding Assistant Directives)

This document contains mandatory guidelines and rules for Gemini Agent, subagents, and future AI coding assistants operating on this codebase.

---

## 1. Mandatory Pre-Flight Checklist

Before making ANY change to the codebase:
1. Read `docs/PROJECT_MASTER_CONTEXT.md` to understand architecture and stack.
2. Read `docs/ENGINE_FREEZE_RULES.md` to check if target files are protected.
3. Inspect source files directly using code search and view tools. NEVER guess variable names, function signatures, or file paths.

---

## 2. Protected Components (STRICT NO-TOUCH)

- **NEVER** modify BiRefNet, Real-ESRGAN, or YuNet ONNX model files or model weights.
- **NEVER** replace BiRefNet with lightweight background removal alternatives without explicit user permission.
- **NEVER** bypass or remove `HEAVY_AI_LOCK` single-inference concurrency protection.
- **NEVER** remove 90-second idle model session unloading or RAM safety limits.
- **NEVER** alter the multi-resolution ICO encoder logic in `src/app/services/image-encoders.ts`.

---

## 3. Modification Rules

1. **Smallest Safe Changes**: Make isolated, targeted edits using precise replacement tools. Avoid sweeping rewrites.
2. **Preserve Documentation & Comments**: Retain existing docstrings and comments.
3. **Runtime Verification**: Never declare a task resolved without running build or test scripts (e.g. `npm run build`, `python backend/tests/test_ico_conversion.py`).
4. **Error Tracebacks First**: Inspect full un-truncated error logs before diagnosing errors.

---

## 4. Standard Response Format for Code Changes

After completing code modifications, always structure your report as follows:

```markdown
### Changed
- File: [path]
- Change: [description]
- Rationale: [why change was made]

### Tested
- Command: [test command]
- Result: [pass/fail status]

### Not Tested
- Item: [item]
- Reason: [reason]

### Risks & Side Effects
- Analysis of potential impacts.

### Documentation Updated
- List of updated docs files.
```
