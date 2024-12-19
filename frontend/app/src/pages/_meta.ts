import { Meta } from 'nextra';

export default {
  "docs": {
    "type": "page",
    "title": "Docs"
  },
  "blog": {
    "type": "page",
    "title": "Blog",
    "theme": {
      "typesetting": "article"
    },
    "display": "hidden"
  },
  "demo": {
    "type": "page",
    "title": "Demo",
    "href": "/",
    "newWindow": true
  }
} satisfies Meta
