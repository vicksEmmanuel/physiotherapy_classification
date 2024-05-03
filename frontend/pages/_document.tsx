import Document, { Head, Html, Main, NextScript } from 'next/document';

const Head2:any = Head;
const NextScript2:any = NextScript;

const Element: React.FC<any> = ({ ...props }) => <Head2 {...props} />;

class MyDocument extends Document {
  render() {
    return (
      <Html lang="en-GB">
        <Element>
          <meta name="description" content="Physio Tees" />

          <meta property="og:title" content="Physio Tees" />
          <meta property="og:site_name" content="Physio Tees" />
          <meta property="og:url" content="https://physio-tees.nelify.app" />
          <meta property="og:description" content="Physio Tees" />
          <meta property="og:type" content="website" />

          <meta property="twitter:card" content="summary_large_image" />
          <meta property="twitter:url" content="https://physio-tees.nelify.app" />
          <meta property="twitter:title" content="Physio Tees" />
          <meta property="twitter:description" content="Physio Tees" />
        </Element>
        <body>
          <Main />
          <NextScript2 />
        </body>
      </Html>
    );
  }
}

export default MyDocument;
