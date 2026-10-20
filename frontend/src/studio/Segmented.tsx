import type { KeyboardEvent } from "react"

export function segmentTabId(group: string, id: string) {
  return `${group}-${id}`.replace(/[^a-zA-Z0-9_-]/g, "")
}

export function Segmented({
  label,
  value,
  options,
  onChange,
  className = "segments",
  mode = "toggle",
  group,
  controls,
  disabled = false,
  hint,
}: {
  label: string
  value: string
  options: { id: string; label: string }[]
  onChange: (id: string) => void
  className?: string
  mode?: "toggle" | "tabs"
  group?: string
  controls?: string
  disabled?: boolean
  hint?: string
}) {
  const tabs = mode === "tabs"

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.altKey || event.metaKey || event.ctrlKey) return
    const keys = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"]
    if (!keys.includes(event.key)) return
    const buttons = [...event.currentTarget.querySelectorAll<HTMLButtonElement>("button:not(:disabled)")]
    if (!buttons.length) return
    const current = buttons.findIndex((button) => button === document.activeElement)
    let next = current < 0 ? 0 : current
    if (event.key === "Home") next = 0
    else if (event.key === "End") next = buttons.length - 1
    else if (event.key === "ArrowRight" || event.key === "ArrowDown") next = (Math.max(current, 0) + 1) % buttons.length
    else next = (Math.max(current, 0) - 1 + buttons.length) % buttons.length
    event.preventDefault()
    const id = buttons[next]?.dataset.value
    if (!id) return
    onChange(id)
    buttons[next]?.focus()
  }

  return (
    <div
      className={className}
      role={tabs ? "tablist" : "group"}
      aria-label={label}
      title={hint}
      onKeyDown={onKeyDown}
    >
      {options.map((option) => {
        const selected = value === option.id
        return (
          <button
            key={option.id}
            type="button"
            data-value={option.id}
            role={tabs ? "tab" : undefined}
            id={tabs && group ? segmentTabId(group, option.id) : undefined}
            aria-selected={tabs ? selected : undefined}
            aria-pressed={tabs ? undefined : selected}
            aria-controls={tabs ? controls : undefined}
            tabIndex={selected ? 0 : -1}
            disabled={disabled}
            onClick={() => onChange(option.id)}
          >
            {option.label}
          </button>
        )
      })}
    </div>
  )
}
