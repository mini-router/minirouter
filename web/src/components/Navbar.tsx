import { Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useEffect, useState } from 'react'
import logoUrl from '../assets/mini-router.jpg'

const links = [
  { to: '/', label: 'Overview' },
  { to: '/leaderboard', label: 'Performance' },
  { to: '/rules', label: 'Rules' },
  { to: '/submit', label: 'Submit' },
]

export default function Navbar() {
  const location = useLocation()
  const [mobileOpen, setMobileOpen] = useState(false)

  useEffect(() => {
    setMobileOpen(false)
  }, [location.pathname])

  return (
    <nav className="fixed inset-x-0 top-0 z-50 glass-strong">
      <div className="section-shell h-16 flex items-center justify-between gap-4">
        <Link to="/" className="flex items-center gap-3">
          <img
            src={logoUrl}
            alt="MiniRouter logo"
            className="h-9 w-9 rounded-xl border border-white/10 bg-white/5 object-cover"
          />
          <span className="leading-tight">
            <span className="block text-sm font-semibold text-text">MiniRouter</span>
            <span className="block text-xs uppercase tracking-[0.22em] text-text-dim">
              Routing
            </span>
          </span>
        </Link>

        <button
          className="md:hidden button-quiet p-2"
          onClick={() => setMobileOpen(!mobileOpen)}
          aria-label="Toggle menu"
          aria-expanded={mobileOpen}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            {mobileOpen ? (
              <path d="M18 6L6 18M6 6l12 12" />
            ) : (
              <path d="M4 6h16M4 12h16M4 18h16" />
            )}
          </svg>
        </button>

        <div className="hidden md:flex items-center gap-2">
          {links.map((link) => (
            <Link
              key={link.to}
              to={link.to}
              className={`rounded-full px-4 py-2 text-sm font-medium transition-colors duration-200 ${
                location.pathname === link.to
                  ? 'bg-white/8 text-text'
                  : 'text-text-dim hover:bg-white/5 hover:text-text'
              }`}
            >
              {link.label}
            </Link>
          ))}
          <span className="ml-2 ui-chip">MiniBridge ready</span>
        </div>
      </div>

      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden border-t border-white/10 overflow-hidden"
          >
            <div className="section-shell py-4 flex flex-col gap-2">
              {links.map((link) => (
                <Link
                  key={link.to}
                  to={link.to}
                  onClick={() => setMobileOpen(false)}
                  className={`rounded-xl px-4 py-3 text-sm font-medium transition-colors ${
                    location.pathname === link.to
                      ? 'bg-white/8 text-text'
                      : 'text-text-dim hover:bg-white/5 hover:text-text'
                  }`}
                >
                  {link.label}
                </Link>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  )
}
