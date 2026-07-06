export default function Footer() {
  return (
    <footer className="border-t border-white/10 py-8">
      <div className="section-shell flex flex-col gap-4 text-sm text-text-dim md:flex-row md:items-center md:justify-between">
        <div>
          <p className="text-text">&copy; {new Date().getFullYear()} MiniRouter Challenge</p>
          <p className="mt-1">
            Built on{' '}
            <a
              href="https://arxiv.org/abs/2512.04695"
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent-light hover:text-accent-light/80"
            >
              TRINITY
            </a>{' '}
            and open-source LLMs
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <span className="ui-chip">GitHub Pages ready</span>
          <span className="ui-chip">Static build</span>
        </div>
      </div>
    </footer>
  )
}
