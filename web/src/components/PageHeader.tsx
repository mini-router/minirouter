import { motion } from 'framer-motion'

interface PageHeaderProps {
  title: string
  subtitle?: string
  eyebrow?: string
}

export default function PageHeader({
  title,
  subtitle,
  eyebrow = 'MiniRouter Challenge',
}: PageHeaderProps) {
  return (
    <section className="section-band pt-28 md:pt-32 pb-14">
      <div className="section-shell">
        <motion.p
          className="section-kicker"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          {eyebrow}
        </motion.p>

        <motion.h1
          className="section-title mt-3 max-w-3xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.08 }}
        >
          {title}
        </motion.h1>

        {subtitle && (
          <motion.p
            className="section-copy mt-4 max-w-2xl"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.16 }}
          >
            {subtitle}
          </motion.p>
        )}

        <div className="divider-line mt-8" />
      </div>
    </section>
  )
}
