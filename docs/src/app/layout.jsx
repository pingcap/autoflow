import 'nextra-theme-docs/style.css'
import './globals.css';

import { Banner, Head } from 'nextra/components'
/* eslint-env node */
import { Footer, Layout, Navbar } from 'nextra-theme-docs'

import { getPageMap } from 'nextra/page-map'

export const metadata = {
  metadataBase: new URL('https://autoflow.ai'),
  title: {
    template: 'AutoFlow'
  },
  description: 'Docs & Blogs of AutoFlow',
  applicationName: 'AutoFlow',
  generator: 'Next.js',
  twitter: {
    site: 'https://twitter.com/tidb_developer'
  }
}

export default async function RootLayout({ children }) {
  const navbar = (
    <Navbar
      logo={
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <svg width="24" height="24" viewBox="0 0 745 745" fill="none" className="logo">
            <rect width="745" height="745" rx="120" className="logo-bg" />
            <rect x="298" y="172" width="150" height="150" rx="24" className="logo-circle" />
            <rect x="298" y="422" width="150" height="150" rx="24" className="logo-circle" />
          </svg>
          <span style={{ marginLeft: '.5em', fontWeight: 300, fontSize: '20px' }}>
            AutoFlow
          </span>
        </div>
      }
      logoLink="/docs"
      projectLink="https://github.com/pingcap/autoflow"
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <a
          target="_blank"
          href="https://twitter.com/tidb_developer"
          aria-label="TiDB Developer Twitter"
          rel="nofollow noreferrer"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="currentColor"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="feather feather-twitter"
          >
            <path d="M23 3a10.9 10.9 0 0 1-3.14 1.53 4.48 4.48 0 0 0-7.86 3v1A10.66 10.66 0 0 1 3 4s-4 9 5 13a11.64 11.64 0 0 1-7 2c9 5 20 0 20-11.5a4.5 4.5 0 0 0-.08-.83A7.72 7.72 0 0 0 23 3z" />
          </svg>
        </a>
        <a
          target="_blank"
          href="https://pingcap.com/ai?utm_source=tidb.ai&utm_medium=community"
          aria-label="TiDB Vector"
          rel="nofollow noreferrer"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 161.24 186.18"
            // viewBox='0 0 24 24'
            fill="currentColor"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="feather feather-tidb"
          >
            <path fill="currentColor" d="M80.62,0L0,46.54v93.09l80.62,46.54,80.62-46.54V46.54L80.62,0ZM80.57,61.98v93.12l-26.77-15.43v-62.24l-26.78,15.46v-30.91l53.54-30.91,26.77,15.45-26.76,15.45ZM134.36,124.12l-26.88,15.52v-62.04l26.88-15.53v62.06Z" />
          </svg>
        </a>
      </div>
    </Navbar>
  )
  const pageMap = await getPageMap()
  return (
    <html lang="en" dir="ltr" suppressHydrationWarning>
      <Head>
        <link
          rel="shortcut icon"
          href="/icon-light.svg"
          type="image/svg+xml"
          media="(prefers-color-scheme: dark)"
        />
        <link
          rel="shortcut icon"
          href="/icon-dark.svg"
          type="image/svg+xml"
          media="(prefers-color-scheme: light)"
        />
      </Head>
      <body>
        <Layout
          navbar={navbar}
          footer={
            <Footer>
              <span>
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 161.24 186.18" className="logo" style={{ width: '24px', height: '24px' }}>
                  <g>
                    <path fill="currentColor" d="M80.62,0L0,46.54v93.09l80.62,46.54,80.62-46.54V46.54L80.62,0ZM80.57,61.98v93.12l-26.77-15.43v-62.24l-26.78,15.46v-30.91l53.54-30.91,26.77,15.45-26.76,15.45ZM134.36,124.12l-26.88,15.52v-62.04l26.88-15.53v62.06Z" />
                  </g>
                </svg>
                <br />
                {new Date().getFullYear()} © <a href="https://pingcap.com" target="_blank" rel="noopener noreferrer">PingCAP</a>. All rights reserved.
              </span>
            </Footer>
          }
          editLink="Edit this page on GitHub"
          docsRepositoryBase="https://github.com/pingcap/autoflow"
          sidebar={{ toggleButton: true, defaultMenuCollapseLevel: 1 }}
          pageMap={pageMap}
        >
          {children}
        </Layout>
      </body>
    </html>
  )
}
