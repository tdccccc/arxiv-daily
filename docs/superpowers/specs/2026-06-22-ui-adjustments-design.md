# UI Adjustments Design

**Date:** 2026-06-22  
**Author:** Claude (brainstorming session)  
**Status:** Draft

## Overview

This design makes several UI adjustments to the arXiv Daily Dashboard and Settings:
1. Adjust play icon size and position
2. Remove Test Connection button (merge into Get Models)
3. Reorganize Settings layout
4. Update Getting Started checklist logic

## Motivation

The current UI has some usability issues:
- Play icon is too large and too close to the border
- Test Connection button is redundant with Get Models
- Settings layout is confusing (Provider, Base URL, API Key, Model order)
- Getting Started checklist doesn't properly track model selection

## Design Decisions

### 1. Play Icon Adjustment

**Current:**
- Size: 10px
- Position: `bottom: 5px; right: 5px`

**New:**
- Size: 8px
- Position: `bottom: 7px; right: 7px`

**Rationale:**
- Smaller icon is less intrusive
- Moving 2px to upper-left avoids border overlap

### 2. Remove Test Connection Button

**Decision:** Remove Test Connection button, merge functionality into Get Models.

**New behavior:**
- Get Models button tests API connectivity implicitly
- Success: "API 连接成功，找到 X 个模型"
- Failure: "API 连接失败：{错误信息}"

**Rationale:**
- If Get Models works → API is connected
- If Get Models fails → API is not connected
- Reduces UI clutter

### 3. Settings Layout Reorganization

**Current order:**
1. Provider
2. API Key
3. Test Connection
4. Base URL
5. Model (dropdown + custom input)
6. Get Models button

**New order:**
1. Base URL (default: DeepSeek URL)
2. API Key
3. Get Models button + Model dropdown (same line)

**Changes:**
- Remove Provider field (not needed)
- Move Base URL above API Key
- Move Model below API Key
- Combine Get Models and Model dropdown on same line
- Remove custom input box for Model
- Set Base URL default to `https://api.deepseek.com/v1`

**Rationale:**
- Provider field is unnecessary (users can manually set Base URL)
- Logical flow: URL → Key → Model
- Get Models and Model dropdown are related, should be together
- Custom input box is redundant (Get Models populates dropdown)

### 4. Getting Started Checklist

**Current:** First item is checked when Base URL and API Key are filled.

**New:** First item requires model selection to be checked.

**Logic:**
- Model dropdown is empty by default
- User must click Get Models to fetch available models
- User must select a model from dropdown
- Only then is the first item checked

**Rationale:**
- Ensures user has actually tested API connectivity
- Ensures user has selected a valid model
- Prevents running with unconfigured model

## Technical Design

### CSS Changes

**File:** `plugin/styles.css`

```css
/* Play icon adjustment */
.arxiv-daily-dashboard__calendar-day.is-runnable .arxiv-daily-dashboard__calendar-day-icon {
  position: absolute;
  bottom: 7px;
  right: 7px;
  width: 8px;
  height: 8px;
  color: var(--color-green);
}

/* Input field width for Base URL and API Key */
.arxiv-daily-settings input[type="text"],
.arxiv-daily-settings input[type="password"] {
  width: 100%;
}
```

### Settings Changes

**File:** `plugin/src/settings/tab.ts`

**Remove:**
- Provider dropdown
- Test Connection button
- Custom model input box

**Add:**
- Base URL default value: `https://api.deepseek.com/v1`
- Get Models button on same line as Model dropdown

**Update:**
- Base URL input width: 100% (longer input field)
- API Key input width: 100% (longer input field)
- Model dropdown: empty by default, populated by Get Models
- Get Models success/failure messages

**Note:** Base URL and API Key input fields should be wider than default to accommodate long URLs and keys.

### Getting Started Changes

**File:** `plugin/src/settings/tab.ts`

**Update checklist logic:**
- First item checked when `s.llm.model` is not empty
- Model dropdown onChange updates checklist

## User Experience Flow

### Settings Configuration

1. User opens Settings
2. Base URL is pre-filled with DeepSeek URL
3. User enters API Key
4. User clicks "Get Models" button
5. If successful:
   - Models appear in dropdown
   - Notice: "API 连接成功，找到 X 个模型"
   - User selects a model
   - Getting Started first item is checked
6. If failed:
   - Notice: "API 连接失败：{错误信息}"
   - User checks Base URL and API Key

### Dashboard Usage

1. User opens Dashboard
2. Calendar shows different states:
   - Green + play icon (8px, positioned at 7px from bottom-right)
   - Purple border + number
   - "0" for no relevant papers
3. User clicks runnable date to generate report

## Implementation Plan

### Phase 1: CSS Changes

1. Update play icon size and position
2. Verify visual appearance

### Phase 2: Settings Layout

1. Remove Provider field
2. Remove Test Connection button
3. Reorder fields: Base URL → API Key → Model
4. Add Get Models button to Model row
5. Remove custom model input
6. Set Base URL default

### Phase 3: Get Models Integration

1. Update Get Models button handler
2. Add success/failure messages
3. Update Model dropdown population

### Phase 4: Getting Started Logic

1. Update checklist first item logic
2. Test model selection flow

## Testing Strategy

### Manual Testing

- Play icon appearance and position
- Settings layout and field order
- Get Models button functionality
- Model dropdown population
- Getting Started checklist updates

### Automated Testing

- Existing tests should still pass
- No new tests needed (UI changes only)

## Open Questions

None - all design decisions have been confirmed with user.

## References

- Current implementation: `plugin/src/settings/tab.ts`
- Current styles: `plugin/styles.css`
- Dashboard view: `plugin/src/dashboard/view.ts`
