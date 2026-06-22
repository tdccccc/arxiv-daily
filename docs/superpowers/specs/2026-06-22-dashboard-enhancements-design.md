# Dashboard Enhancements Design

**Date:** 2026-06-22  
**Author:** Claude (brainstorming session)  
**Status:** Draft

## Overview

This design enhances the arXiv Daily Dashboard with two features:
1. **Settings button** - Quick access to plugin settings from the Dashboard
2. **Runnable dates in calendar** - Visual indicators on dates that can generate daily reports

## Motivation

Users currently need to navigate through Obsidian's settings to configure the plugin. Adding a settings button directly in the Dashboard improves accessibility.

The calendar currently shows dates with existing daily reports but doesn't indicate which dates could have reports generated. Adding "runnable date" markers allows users to quickly identify and run reports for missing dates.

## Design Decisions

### 1. Settings Button Placement

**Decision:** Place the settings button in the toolbar actions area, next to the "More" button (right side).

**Rationale:**
- Consistent with existing toolbar button placement
- Easy to discover without cluttering the main view
- Follows Obsidian's UI patterns (settings access from toolbar)

### 2. Calendar Runnable Dates

**Decision:** Extend the calendar model with a unified date state system.

**Date States:**
- `has-report` - Date has an existing daily report (purple border, current behavior)
- `runnable` - Date is within lookback window, no report, can be run (green background, play icon)
- `running` - Date is currently being processed (future extension)
- `failed` - Date failed to generate report (future extension)
- `empty` - Date outside lookback window or no data available

**Visual Design for Runnable Dates:**
- Light green semi-transparent background (using `::before` pseudo-element with `var(--color-green)` at 15% opacity)
- Green border (`var(--color-green)`)
- Play icon (▶) in the date cell
- Hover tooltip: "Click to run for YYYY-MM-DD"
- Click handler: Trigger `runForDate` command

## Technical Design

### Architecture Changes

#### 1. Calendar State Model

Add a new type to represent calendar cell state:

```typescript
type CalendarCellState = 
  | "empty"        // No date or outside lookback
  | "runnable"     // Can generate report
  | "has-report"   // Report exists
  | "running"      // Currently processing
  | "failed"       // Failed to generate
  | "skipped";     // Intentionally skipped
```

#### 2. Calendar Cell Interface

Extend the calendar cell data structure:

```typescript
interface CalendarCell {
  date: string | null;
  state: CalendarCellState;
  report?: DailyReportDay;
  runState?: RunStateEntry;
}
```

#### 3. Runnable Date Detection

A date is considered "runnable" if:
1. Date is within the lookback window (5 days by default)
2. No daily report exists for that date
3. Run state is `pending` or `failed_transient` (not `running`, `completed`, `failed_permanent`, or `skipped`)
4. Date is not a weekend (optional, based on settings)

### Component Changes

#### 1. `renderHeader` Method

**Current:**
```typescript
private renderHeader(contentEl: HTMLElement): void {
  const header = contentEl.createEl("div", {
    cls: "arxiv-daily-dashboard__header",
  });
  header.createEl("h2", { text: "arXiv Daily Dashboard" });
}
```

**New:**
```typescript
private renderHeader(contentEl: HTMLElement): void {
  const header = contentEl.createEl("div", {
    cls: "arxiv-daily-dashboard__header",
  });
  header.createEl("h2", { text: "arXiv Daily Dashboard" });
  
  // Add settings button
  const actions = header.createEl("div", {
    cls: "arxiv-daily-dashboard__header-actions",
  });
  this.createSettingsButton(actions);
}
```

#### 2. New `createSettingsButton` Method

```typescript
private createSettingsButton(parent: HTMLElement): void {
  const button = parent.createEl("button", {
    cls: "arxiv-daily-dashboard__settings-btn",
    attr: {
      type: "button",
      "aria-label": "Open arXiv Daily settings",
    },
  }) as HTMLButtonElement;
  setIcon(button, "settings");
  button.createSpan({ text: "Settings" });
  button.addEventListener("click", () => this.openSettings());
}
```

#### 3. `renderDailyCalendar` Method

**Changes:**
- Add logic to determine calendar cell state
- Apply different CSS classes based on state
- Add click handlers for runnable dates

```typescript
private renderDailyCalendar(contentEl: HTMLElement): void {
  // ... existing code ...
  
  for (const cell of this.buildCalendarCells(month)) {
    const button = grid.createEl("button", {
      cls: this.getCalendarCellClasses(cell),
      attr: {
        type: "button",
        "aria-label": this.getCalendarCellAriaLabel(cell),
      },
    }) as HTMLButtonElement;
    
    if (!cell.date) {
      button.disabled = true;
      button.addClass("is-empty");
      continue;
    }
    
    // Date number
    button.createSpan({
      cls: "arxiv-daily-dashboard__calendar-day-number",
      text: String(Number(cell.date.slice(-2))),
    });
    
    // Today indicator
    if (cell.date === today) button.addClass("is-today");
    
    // State-specific rendering
    switch (cell.state) {
      case "has-report":
        this.renderReportCell(button, cell);
        break;
      case "runnable":
        this.renderRunnableCell(button, cell);
        break;
      case "running":
        this.renderRunningCell(button, cell);
        break;
      case "failed":
        this.renderFailedCell(button, cell);
        break;
    }
  }
}
```

#### 4. New `buildCalendarCells` Method

```typescript
private buildCalendarCells(month: string): CalendarCell[] {
  const cells: CalendarCell[] = [];
  const byDate = new Map(this.dailyReports.map(r => [r.date, r]));
  const today = this.todayDate();
  const lookbackDates = this.getLookbackDates();
  
  for (const cellDate of calendarCells(month)) {
    if (!cellDate.date) {
      cells.push({ date: null, state: "empty" });
      continue;
    }
    
    const report = byDate.get(cellDate.date);
    const runState = this.plugin.stateStore.get(cellDate.date);
    
    let state: CalendarCellState;
    if (report) {
      state = "has-report";
    } else if (lookbackDates.has(cellDate.date)) {
      if (runState.status === "running") {
        state = "running";
      } else if (runState.status === "failed_permanent" || runState.status === "skipped") {
        state = "failed";
      } else {
        state = "runnable";
      }
    } else {
      state = "empty";
    }
    
    cells.push({
      date: cellDate.date,
      state,
      report,
      runState,
    });
  }
  
  return cells;
}
```

#### 5. New `renderRunnableCell` Method

```typescript
private renderRunnableCell(button: HTMLButtonElement, cell: CalendarCell): void {
  button.addClass("is-runnable");
  
  // Play icon
  const icon = button.createSpan({
    cls: "arxiv-daily-dashboard__calendar-day-icon",
  });
  setIcon(icon, "play");
  
  // Click handler to run
  button.addEventListener("click", () => {
    void this.runDateFromCalendar(cell.date!);
  });
}
```

#### 6. New `runDateFromCalendar` Method

```typescript
private async runDateFromCalendar(date: string): Promise<void> {
  const setup = getSetupStatus(this.plugin.settings);
  if (!setup.readyToRun) {
    new Notice("arXiv Daily: Please complete setup first");
    this.openSettings();
    return;
  }
  
  new Notice(`arXiv Daily: running for ${date}…`);
  const result = await this.plugin.scheduler.runForDateNow(date);
  new Notice(`arXiv Daily ${date}: ${describeResult(result)}`);
  
  // Refresh dashboard
  await this.reloadIndex();
}
```

### CSS Changes

#### 1. Settings Button

```css
.arxiv-daily-dashboard__header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.arxiv-daily-dashboard__header-actions {
  display: flex;
  gap: 8px;
}

.arxiv-daily-dashboard__settings-btn {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 8px;
  border-radius: 4px;
  background: var(--interactive-normal);
  color: var(--text-normal);
  cursor: pointer;
}

.arxiv-daily-dashboard__settings-btn:hover {
  background: var(--interactive-hover);
}
```

#### 2. Runnable Date Cell

```css
.arxiv-daily-dashboard__calendar-day.is-runnable {
  background-color: transparent;
  border: 1px solid var(--color-green);
  cursor: pointer;
  position: relative;
}

.arxiv-daily-dashboard__calendar-day.is-runnable::before {
  content: "";
  position: absolute;
  inset: 0;
  background-color: var(--color-green);
  opacity: 0.15;
  border-radius: inherit;
  z-index: -1;
}

.arxiv-daily-dashboard__calendar-day.is-runnable:hover::before {
  opacity: 0.3;
}

.arxiv-daily-dashboard__calendar-day.is-runnable:hover {
  box-shadow: 0 0 0 1px var(--color-green);
}

.arxiv-daily-dashboard__calendar-day.is-runnable .arxiv-daily-dashboard__calendar-day-icon {
  position: absolute;
  top: 2px;
  right: 2px;
  width: 12px;
  height: 12px;
  color: var(--color-green);
}

.arxiv-daily-dashboard__calendar-day.is-runnable::after {
  content: attr(aria-label);
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%);
  padding: 4px 8px;
  background: var(--background-modifier-tooltip);
  color: var(--text-normal);
  border-radius: 4px;
  font-size: 12px;
  white-space: nowrap;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.2s;
  z-index: 10;
}

.arxiv-daily-dashboard__calendar-day.is-runnable:hover::after {
  opacity: 1;
}
```

### Future Extensions

The state-based design allows easy addition of:

1. **Running State:**
   - Visual: Spinning icon, animated border
   - Click: Show progress or cancel option

2. **Failed State:**
   - Visual: Red background, warning icon
   - Click: Show error details, retry option

3. **Skipped State:**
   - Visual: Grayed out, skip icon
   - Click: Unskip option

## User Experience Flow

### Settings Button
1. User opens Dashboard
2. Clicks "Settings" button in top-right corner
3. Plugin settings tab opens immediately

### Runnable Date
1. User sees calendar with green-highlighted dates
2. Hovers over date to see "Click to run for YYYY-MM-DD"
3. Clicks date to trigger report generation
4. Dashboard refreshes to show new report

## Implementation Plan

### Phase 1: Settings Button
1. Modify `renderHeader` to include settings button
2. Add CSS for button styling
3. Test settings navigation

### Phase 2: Calendar State Model
1. Define `CalendarCellState` type
2. Implement `buildCalendarCells` method
3. Add state detection logic

### Phase 3: Runnable Date Rendering
1. Add CSS classes for runnable state
2. Implement `renderRunnableCell` method
3. Add click handler for running dates

### Phase 4: Integration & Testing
1. Integrate with existing scheduler
2. Test edge cases (weekends, lookback window)
3. Verify accessibility (keyboard navigation, screen readers)

## Testing Strategy

### Unit Tests
- Calendar cell state detection
- Runnable date identification
- Settings button click handler

### Integration Tests
- Dashboard refresh after running date
- Settings navigation from Dashboard
- Edge cases: weekends, lookback boundaries

### Manual Testing
- Visual verification of runnable dates
- Click behavior on different date states
- Settings button functionality

## Design Decisions (Resolved)

### Weekend Handling
**Decision:** Weekends are **not** marked as runnable. The scheduler already skips weekends (line 86 in scheduler.ts: `if (isWeekendDate(dateObj)) continue;`), so the calendar should reflect this behavior.

**Rationale:**
- Consistent with existing scheduler behavior
- Avoids confusion when users click a weekend date and nothing happens
- arXiv typically doesn't publish new papers on weekends

### Lookback Window
**Decision:** Use the hardcoded `LOOKBACK_DAYS = 5` constant from scheduler.ts. This is not configurable in settings.

**Rationale:**
- Matches the scheduler's behavior exactly
- Keeps the feature simple for initial implementation
- Can be made configurable later if needed

### Visual Feedback for Running Dates
**Decision:** Running dates will show a spinning icon (future extension). For the initial implementation, only `runnable` and `has-report` states are fully implemented.

**Rationale:**
- Focuses scope on the core feature
- Running state requires real-time updates which add complexity
- Can be added in a future iteration

### Error Handling for Calendar Clicks
**Decision:** If a run fails, the date will remain in `runnable` state (if `failed_transient`) or move to `failed` state (if `failed_permanent`). Users can click again to retry.

**Rationale:**
- Matches the scheduler's retry logic
- Provides clear visual feedback about failure state
- Allows users to manually retry failed dates

## References

- Current implementation: `plugin/src/dashboard/view.ts`
- Scheduler: `plugin/src/services/scheduler.ts`
- State store: `plugin/src/services/state-store.ts`
- Settings types: `plugin/src/settings/types.ts`
