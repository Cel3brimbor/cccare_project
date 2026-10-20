import { useEffect, useRef, type RefObject } from "react"

function focusable(root: HTMLElement) {
  return [...root.querySelectorAll<HTMLElement>("button, a[href], input, textarea, select")].filter((element) => {
    if (element.hasAttribute("disabled")) return false
    if (element.getAttribute("aria-hidden") === "true") return false
    return true
  })
}

export function useDialog<T extends HTMLElement>(open: boolean, ref: RefObject<T | null>, onClose: () => void) {
  const closeRef = useRef(onClose)
  closeRef.current = onClose

  useEffect(() => {
    if (!open) return
    const root = ref.current
    if (!root) return
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const initial = root.querySelector<HTMLElement>("[data-initial-focus]")
    if (initial && !initial.hasAttribute("disabled")) initial.focus({ preventScroll: true })
    else root.focus({ preventScroll: true })

    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault()
        closeRef.current()
        return
      }
      if (event.key !== "Tab") return
      const items = focusable(root)
      if (!items.length) {
        event.preventDefault()
        return
      }
      const first = items[0]
      const last = items[items.length - 1]
      const active = document.activeElement
      const inside = active instanceof Node && root.contains(active)
      if (event.shiftKey && (!inside || active === first || active === root)) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && (!inside || active === last)) {
        event.preventDefault()
        first.focus()
      }
    }
    document.addEventListener("keydown", onKey)
    return () => {
      document.removeEventListener("keydown", onKey)
      if (previous && document.contains(previous)) previous.focus({ preventScroll: true })
    }
  }, [open, ref])
}
