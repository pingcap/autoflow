import nextra from 'nextra'

const withNextra = nextra({
  latex: true,
  defaultShowCopyCode: true,
  search: {
    codeblocks: true
  },
  contentDirBasePath: '/',
})

export default withNextra({
  // reactStrictMode: true
})
