import katex from "katex"
import { memo, useMemo } from "react"

export const Formula = memo(function Formula({ tex }: { tex: string }) {
  const html = useMemo(() => katex.renderToString(tex, { throwOnError: false, displayMode: true }), [tex])
  return <div className="formula" dangerouslySetInnerHTML={{ __html: html }} />
})
