# Pipeline Error Handling & Calendar Display Design

**Date:** 2026-06-22  
**Author:** Claude (brainstorming session)  
**Status:** Draft

## Overview

This design improves the pipeline error handling and calendar display logic for the arXiv Daily Dashboard. The core principle is: **don't write empty files** - instead, let the scheduler retry later or show appropriate status in the calendar.

## Motivation

The current pipeline writes empty daily files when:
1. arXiv returns 0 papers
2. LLM filtering results in 0 relevant papers
3. All papers are ignored

This causes problems:
- Empty files prevent future retries (scheduler skips dates with existing files)
- Users can't distinguish between "arXiv didn't update" and "no relevant papers"
- Configuration/API errors are silently swallowed

## Design Decisions

### 1. Pipeline Error Handling

**Decision:** Don't write empty files. Let the scheduler retry for transient issues.

**New behavior:**

| Stage | Situation | Old Behavior | New Behavior |
|-------|-----------|--------------|--------------|
| arXiv fetch | 0 papers returned | Write empty file, completed | Don't write file, let scheduler retry |
| LLM filter | 0 relevant papers | Write empty file, completed | Don't write file, show "0" in calendar |
| Index | 0 visible papers | Write empty file, completed | Don't write file, show "0" in calendar |
| Any stage | Network/API error | Write empty file, completed | Don't write file, let scheduler retry |
| Any stage | Success | Write file with content | Write file with content |

**Rationale:**
- arXiv may not have updated yet (retry later)
- LLM filtering returning 0 papers could be configuration issue
- Network errors are transient (retry later)
- Only write files when we have actual content

### 2. Calendar Display Logic

**Decision:** Show different states based on the situation.

**Display states:**

| Situation | Calendar Display | Tooltip |
|-----------|-----------------|---------|
| Weekend | Normal (no mark) | arXiv 未更新 |
| Weekday, arXiv not updated | Normal (no mark) | arXiv 未更新 |
| Weekday, 0 papers from LLM | Show "0" | 0 篇相关论文 |
| Weekday, successful report | Purple border + count | Open daily report... |
| Runnable | Green + play icon | Click to run for... |

**Note:** "0 篇相关论文" means arXiv had papers but none were relevant after LLM filtering. This is different from "arXiv 未更新" which means arXiv hasn't published papers yet.

**Rationale:**
- Users can see at a glance what happened for each date
- Different tooltips provide context
- "0" display helps identify dates with no relevant papers

### 3. Runnable State Logic

**Decision:** A date is "runnable" if the scheduler would try to run it on the next tick.

**Conditions:**

```typescript
function isRunnable(
  date: string,
  today: string,
  hasFile: boolean,
  startTime: number,
  endTime: number,
  currentTime: number
): boolean {
  // In 5-day lookback window
  if (!isInLookbackWindow(date)) return false;
  
  // Not weekend
  if (isWeekend(date)) return false;
  
  // No local daily file
  if (hasFile) return false;
  
  // If today, must be within running time window
  if (date === today) {
    return currentTime >= startTime && currentTime < endTime;
  }
  
  // Past dates: runnable if in lookback and no file
  return true;
}
```

**Time window for today:**
- Before start time (e.g., 09:00): NOT runnable (arXiv hasn't updated)
- Start time to end time (e.g., 09:00-18:00): RUNNABLE (if no file)
- After end time (e.g., 18:00): NOT runnable (arXiv probably won't update)

**Past dates (within lookback window):**
- If no file exists: RUNNABLE (scheduler will try to fetch)
- If file exists: NOT runnable (already processed)
- Note: If arXiv skipped a date (no papers), the scheduler will fetch 0 papers and return `pending`. On the next day, if arXiv publishes papers for that date, the scheduler will fetch them. If arXiv doesn't publish, the scheduler will keep retrying until the lookback window passes.

**Rationale:**
- Matches scheduler behavior
- Prevents unnecessary retries outside active hours
- Clear logic for users to understand
- Handles the case where arXiv skips a day and publishes later

### 4. API Connectivity Testing

**Decision:** Add basic connectivity testing for arXiv and LLM APIs.

**Features:**
- Test arXiv API connectivity
- Test LLM API connectivity
- Manual trigger via "Test Connection" button
- Auto-trigger on configuration change
- Show available models in dropdown

**UI placement:**
- In Settings tab, near API configuration
- "Test Connection" button next to API fields
- Auto-test when API key/URL changes

**Rationale:**
- Helps users diagnose configuration issues early
- Prevents silent failures
- Improves user experience

### 5. Model Listing

**Decision:** Fetch available models from API using OpenAI-compatible endpoint.

**Implementation:**
- Use `GET /v1/models` endpoint (OpenAI-compatible)
- Try multiple candidate URLs (like cc-switch)
- 15-second timeout per request
- Show models in dropdown for selection

**Candidate URL logic:**
1. `baseURL + /v1/models`
2. If baseURL ends with known suffix (e.g., `/anthropic`), strip and try again
3. Try `baseURL + /models` as fallback

**Error handling:**
- Show specific error messages (auth failed, endpoint not found, timeout)
- If API doesn't support listing models, show error and suggest manual input

**Rationale:**
- Most APIs support OpenAI-compatible `/v1/models`
- Candidate URL approach handles various URL formats
- Clear error messages help users fix configuration

## Technical Design

### Pipeline Changes

#### 1. `runForDateInner` Method

**Current behavior:**
- arXiv returns 0 papers → write empty file, return `completed`
- LLM filter returns 0 papers → write empty file, return `completed`
- Index returns 0 papers → write empty file, return `completed`

**New behavior:**
- arXiv returns 0 papers → return `pending` (don't write file)
- LLM filter returns 0 papers → return `completed` with `papersWritten: 0` (don't write file)
- Index returns 0 papers → return `completed` with `papersWritten: 0` (don't write file)

**Code changes:**

```typescript
// arXiv returns 0 papers
if (sourcePapers.length === 0) {
  throwIfCancelled(signal);
  // Don't write empty file - let scheduler retry later
  return { kind: "pending", reason: "no papers from arXiv" };
}

// LLM filter returns 0 papers
if (filtered.length === 0) {
  throwIfCancelled(signal);
  // Don't write empty file - show "0" in calendar
  return { kind: "completed", papersWritten: 0 };
}

// Index returns 0 papers
if (visiblePapers.length === 0) {
  throwIfCancelled(signal);
  // Don't write empty file - show "0" in calendar
  return { kind: "completed", papersWritten: 0 };
}
```

#### 2. PipelineResult Type

**Add new result kind:**

```typescript
export type PipelineResult =
  | { kind: "completed"; papersWritten: number }
  | { kind: "pending"; reason: string }
  | { kind: "failed_transient"; reason: string }
  | { kind: "failed_permanent"; reason: string };
```

#### 3. Scheduler Integration

**Update scheduler to handle `pending` result:**

```typescript
// In tryRun method
if (result.kind === "completed") {
  await this.deps.store.setCompleted(date, result.papersWritten);
} else if (result.kind === "pending") {
  // Don't mark as completed - let scheduler retry
  this.deps.logger.info(`arXiv ${date}: pending - ${result.reason}`);
} else if (result.kind === "failed_transient") {
  await this.deps.store.setFailed(date, "transient", result.reason);
} else {
  await this.deps.store.setFailed(date, "permanent", result.reason);
}
```

### Calendar Changes

#### 1. Calendar Cell State

**Extend CalendarCellState:**

```typescript
type CalendarCellState = 
  | "empty"        // No date or outside lookback
  | "runnable"     // Can generate report
  | "has-report"   // Report exists
  | "no-papers";   // LLM filtered to 0 papers
```

#### 2. Calendar Cell Builder

**Update buildCalendarCells:**

```typescript
private buildCalendarCells(month: string): CalendarCell[] {
  const cells: CalendarCell[] = [];
  const byDate = new Map(this.dailyReports.map(r => [r.date, r]));
  const lookbackDates = this.getLookbackDates();

  for (const cellDate of calendarCells(month)) {
    if (!cellDate.date) {
      cells.push({ date: null, state: "empty" });
      continue;
    }

    const report = byDate.get(cellDate.date);
    const hasFile = report !== undefined;

    let state: CalendarCellState;
    if (report) {
      state = report.papers === 0 ? "no-papers" : "has-report";
    } else if (this.isRunnable(cellDate.date)) {
      state = "runnable";
    } else if (isWeekendDate(new Date(cellDate.date))) {
      state = "empty"; // Weekend
    } else {
      state = "empty"; // arXiv not updated or outside lookback
    }

    cells.push({
      date: cellDate.date,
      state,
      report,
    });
  }

  return cells;
}
```

#### 3. Runnable Detection

**Implement isRunnable method:**

```typescript
private isRunnable(date: string): boolean {
  const today = this.todayDate();
  const hasFile = this.dailyReports.some(r => r.date === date);
  
  // In lookback window
  if (!this.getLookbackDates().has(date)) return false;
  
  // Not weekend
  if (isWeekendDate(new Date(date))) return false;
  
  // No file
  if (hasFile) return false;
  
  // If today, check time window
  if (date === today) {
    const now = new Date();
    const settings = this.plugin.settings;
    const startTime = parseHHMM(settings.schedule.runAtLocal);
    const endTime = { hour: 18, minute: 0 }; // Default end time
    
    const currentMinutes = now.getHours() * 60 + now.getMinutes();
    const startMinutes = startTime.hour * 60 + startTime.minute;
    const endMinutes = endTime.hour * 60 + endTime.minute;
    
    return currentMinutes >= startMinutes && currentMinutes < endMinutes;
  }
  
  // Past dates: runnable
  return true;
}
```

#### 4. Calendar Rendering

**Update renderCalendarCell:**

```typescript
private renderCalendarCell(cell: CalendarCell): HTMLElement {
  const el = document.createElement("div");
  el.className = "arxiv-daily-dashboard__calendar-cell";

  switch (cell.state) {
    case "has-report":
      el.addClass("has-report");
      el.textContent = String(cell.report?.papers ?? 0);
      el.setAttribute("aria-label", `Open daily report ${cell.date}`);
      break;
      
    case "no-papers":
      el.addClass("no-papers");
      el.textContent = "0";
      el.setAttribute("aria-label", "0 篇相关论文");
      break;
      
    case "runnable":
      el.addClass("is-runnable");
      el.innerHTML = '<span class="play-icon">▶</span>';
      el.setAttribute("aria-label", `Click to run for ${cell.date}`);
      break;
      
    case "empty":
    default:
      // Check if weekend or arXiv not updated
      if (isWeekendDate(new Date(cell.date ?? ""))) {
        el.setAttribute("aria-label", "arXiv 未更新");
      } else {
        el.setAttribute("aria-label", "arXiv 未更新");
      }
      break;
  }

  return el;
}
```

### Settings Changes

#### 1. API Connectivity Test

**Add test connection button:**

```typescript
// In settings tab
const testButton = containerEl.createEl("button", {
  text: "Test Connection",
  cls: "arxiv-daily-settings__test-btn",
});

testButton.addEventListener("click", async () => {
  testButton.disabled = true;
  testButton.textContent = "Testing...";
  
  try {
    const result = await this.testApiConnection();
    if (result.success) {
      new Notice("API connection successful!");
    } else {
      new Notice(`API connection failed: ${result.error}`);
    }
  } catch (e) {
    new Notice(`API connection failed: ${(e as Error).message}`);
  } finally {
    testButton.disabled = false;
    testButton.textContent = "Test Connection";
  }
});
```

#### 2. Model Listing

**Add model fetch button:**

```typescript
// In settings tab, near model input
const fetchModelsButton = containerEl.createEl("button", {
  text: "Get Models",
  cls: "arxiv-daily-settings__fetch-models-btn",
});

fetchModelsButton.addEventListener("click", async () => {
  fetchModelsButton.disabled = true;
  fetchModelsButton.textContent = "Fetching...";
  
  try {
    const models = await this.fetchAvailableModels();
    this.showModelDropdown(models);
  } catch (e) {
    new Notice(`Failed to fetch models: ${(e as Error).message}`);
  } finally {
    fetchModelsButton.disabled = false;
    fetchModelsButton.textContent = "Get Models";
  }
});
```

**Implement fetchAvailableModels:**

```typescript
private async fetchAvailableModels(): Promise<string[]> {
  const settings = this.plugin.settings.llm;
  const baseUrl = settings.baseUrl.replace(/\/+$/, "");
  const apiKey = settings.apiKey;
  
  if (!baseUrl || !apiKey) {
    throw new Error("Please fill in API Base URL and API Key first");
  }
  
  // Try multiple candidate URLs
  const candidates = this.buildModelUrlCandidates(baseUrl);
  
  for (const url of candidates) {
    try {
      const response = await this.plugin.app.vault.adapter.fetch(url, {
        method: "GET",
        headers: {
          "Authorization": `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        return this.parseModelList(data);
      }
    } catch (e) {
      // Try next candidate
      continue;
    }
  }
  
  throw new Error("Failed to fetch models from any endpoint");
}

private buildModelUrlCandidates(baseUrl: string): string[] {
  const candidates: string[] = [];
  
  // Primary: baseURL + /v1/models
  candidates.push(`${baseUrl}/v1/models`);
  
  // If URL ends with known suffix, strip and try
  const knownSuffixes = [
    "/api/claudecode",
    "/api/anthropic",
    "/apps/anthropic",
    "/api/coding",
    "/claudecode",
    "/anthropic",
    "/step_plan",
    "/coding",
    "/claude",
  ];
  
  for (const suffix of knownSuffixes) {
    if (baseUrl.endsWith(suffix)) {
      const stripped = baseUrl.slice(0, -suffix.length);
      candidates.push(`${stripped}/v1/models`);
      break;
    }
  }
  
  // Fallback: baseURL + /models
  candidates.push(`${baseUrl}/models`);
  
  return candidates;
}

private parseModelList(data: any): string[] {
  if (data.data && Array.isArray(data.data)) {
    return data.data.map((model: any) => model.id).filter(Boolean);
  }
  if (Array.isArray(data)) {
    return data.map((model: any) => model.id || model.name).filter(Boolean);
  }
  throw new Error("Invalid model list format");
}
```

## User Experience Flow

### Calendar Display

1. User opens Dashboard
2. Calendar shows different states for each date:
   - Purple border + number: successful report
   - "0": no relevant papers
   - Green + play icon: runnable
   - No mark: arXiv not updated or weekend
3. Hover over date to see tooltip with context

### API Testing

1. User fills in API Base URL and API Key
2. Clicks "Test Connection" button
3. Sees success/error message
4. Clicks "Get Models" button
5. Sees dropdown with available models
6. Selects model from dropdown

## Implementation Plan

### Phase 1: Pipeline Error Handling

1. Update pipeline to not write empty files
2. Add `pending` result kind
3. Update scheduler to handle `pending` result
4. Test error scenarios

### Phase 2: Calendar Display

1. Add `no-papers` state to CalendarCellState
2. Update calendar cell builder
3. Update calendar rendering
4. Add tooltips for different states

### Phase 3: Runnable Logic

1. Implement isRunnable method
2. Add time window checking
3. Update calendar to show runnable state
4. Test edge cases

### Phase 4: API Testing

1. Add test connection button
2. Implement API connectivity test
3. Add error handling and messages
4. Test with different API providers

### Phase 5: Model Listing

1. Implement fetchAvailableModels
2. Add candidate URL logic
3. Add model dropdown UI
4. Test with different APIs

## Testing Strategy

### Unit Tests

- Pipeline error handling
- Calendar cell state logic
- Runnable detection
- Model URL candidate building

### Integration Tests

- End-to-end pipeline flow
- Calendar display with different states
- API testing with mock responses

### Manual Testing

- Calendar display with various scenarios
- API testing with real APIs
- Model listing with different providers

## Design Decisions (Resolved)

### End Time Configuration
**Decision:** The end time (18:00) should be configurable in settings, with a default of 18:00.

**Rationale:**
- Different users may have different schedules
- arXiv update times may vary by timezone
- Allows users to customize when the scheduler stops retrying

### Retry Interval
**Decision:** Use the existing `tickIntervalMin` setting (default: 20 minutes) for retry interval.

**Rationale:**
- Already configurable in settings
- Consistent with current scheduler behavior

### Model List Caching
**Decision:** Don't cache the model list. Fetch fresh list each time user clicks "Get Models".

**Rationale:**
- Model lists don't change frequently
- Simpler implementation
- Users can manually refresh if needed

### Error Message Localization
**Decision:** Use English for error messages initially. Can be localized later if needed.

**Rationale:**
- Faster implementation
- Consistent with existing codebase (most messages are in English)
- Can add localization later without breaking changes

## References

- Current implementation: `plugin/src/pipeline/pipeline.ts`
- Scheduler: `plugin/src/services/scheduler.ts`
- Calendar: `plugin/src/dashboard/view.ts`
- Settings: `plugin/src/settings/tab.ts`
- cc-switch implementation: https://github.com/farion1231/cc-switch
