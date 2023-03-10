import { useState, useEffect, useRef } from "react";

import Highlighter from "react-highlight-words";

import Head from "next/head";
import Image from "next/image";

export default function Home() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([
    "Start by searching for matching trials... try 'cancer' or 'asthma' or 'oxygen saturation' ... and click the search button.",
  ]);
  const [isLoading, setLoading] = useState(true);

  useEffect(() => {
    if (query !== "") {
      // setLoading(true);
      let url = "/api/forwardme?query=" + query;
      let response = fetch(url);
      response.then(async (res) => {
        res = await res.text();
        // console.log({ res });
        // res.split("####");
        setResults(res.split("####"));
        setLoading(false);
      });
    }
  }, [query]);

  function search() {
    setQuery(document.getElementById("searchFor").value);
  }

  function keypress(event) {
    // If the user presses the "Enter" key on the keyboard
    if (event.key === "Enter") {
      search();
    }
  }

  const focusme = useRef(null);

  useEffect(() => {
    if (focusme.current) {
      focusme.current.focus();
    }
  }, []);

  return (
    <div>
      <Head>

<title>Search 15k clinical trials using machine learning - medBERT/NLP demo</title>
<meta name="title" content="Search 15k clinical trials using machine learning - medBERT/NLP demo"/>
<meta name="description" content="Try it out by searching above for matching trials... try 'cancer' or 'asthma' or 'oxygen saturation' ... and click the search button. Note that some accurate matches do not contain the keywords you searched for, because language inference is taking the place of keyword search."/>


<meta property="og:type" content="website"/>
<meta property="og:url" content="https://machine-learning-demo.vercel.app/"/>
<meta property="og:title" content="Search 15k clinical trials using machine learning - medBERT/NLP demo"/>
<meta property="og:description" content="Try it out by searching above for matching trials... try 'cancer' or 'asthma' or 'oxygen saturation' ... and click the search button. Note that some accurate matches do not contain the keywords you searched for, because language inference is taking the place of keyword search."/>
<meta property="og:image" content="https://raw.githubusercontent.com/zachblume/machine-learning-recommendation-engine-demo/main/newscreenshot.png"/>


    <meta property="twitter:card" content="summary_large_image"/>
<meta property="twitter:url" content="https://machine-learning-demo.vercel.app/"/>
<meta property="twitter:title" content="Search 15k clinical trials using machine learning - medBERT/NLP demo"/>
<meta property="twitter:description" content="Try it out by searching above for matching trials... try 'cancer' or 'asthma' or 'oxygen saturation' ... and click the search button. Note that some accurate matches do not contain the keywords you searched for, because language inference is taking the place of keyword search."/>
<meta property="twitter:image" content="https://raw.githubusercontent.com/zachblume/machine-learning-recommendation-engine-demo/main/newscreenshot.png"/>
      </Head>

      <main>
        <h1 className="text-3xl font-bold">
          Search 15k clinical trials using machine learning - medBERT/NLP demo
        </h1>

        <input
          type="text"
          id="searchFor"
          className=""
          onKeyDown={keypress}
          onChange={search}
          ref={focusme}
        />
        <input
          type="submit"
          value="Search"
          onClick={search}
          id="submit"
          onKeyDown={keypress}
        />

        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3 text-sm">
          <div className="mb-3 col-span-1 divide-y divide-gray-200 rounded-lg bg-white shadow p-3 px-4">
            <b>Try it out</b> by searching above for matching trials... try
            &apos;cancer&apos; or &apos;asthma&apos; or &apos;oxygen
            saturation&apos; ... and click the search button.{" "}
            <i className="border-none">
              Note that some accurate matches do not contain the keywords you
              searched for, because language inference is taking the place of
              keyword search.
            </i>
          </div>

          <div className="col-span-1 divide-y divide-gray-200 rounded-lg bg-white shadow  p-3 px-4">
            The backend is a Python web server running on a Google Compute
            instance, using machine learning (medBERT, an implementation of
            Google&apos;s BERT transformer) to pre-compute vector embeddings for
            the text abstracts taken from a database of public clinical trials,
            and compare the current query against that vector space, returning a
            dot-product distance &apos;score&apos;.
          </div>
          <div className="mb-1 col-span-1 divide-y divide-gray-200 rounded-lg bg-white shadow p-3 px-4">
            This frontend is a Next.JS app running on Vercel. Take a look at the
            code on github at{" "}
            <a
              href="https://github.com/zachblume/machine-learning-recommendation-engine-demo"
              className="font-medium text-blue-600 dark:text-blue-500 "
            >
              https://github.com/zachblume/machine-learning-recommendation-engine-demo
            </a>
          </div>
        </div>
        <br />
        <ul
          role="list"
          className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4"
        >
          {isLoading
            ? "Type something in the search box at the top to begin..."
            : results.map((result, index) => {
                let resultSplit = result.split("||||");
                let truncatedText =
                  (resultSplit[1] || "").substring(9, 250) || "";
                let score;
                if (resultSplit.length > 1) {
                  score = resultSplit[0] || 0;
                  score = parseFloat(score).toFixed(4) || 0;
                } else score = resultSplit[0];

                return (
                  <li
                    className="col-span-1 divide-y divide-gray-200 rounded-lg bg-white shadow"
                    key={index}
                  >
                    <div className="flex w-full items-center justify-between space-x-6 p-6">
                      <div className="flex-1">
                        <p className="mt-1  text-sm text-gray-500">
                          {resultSplit.length > 1 ? (
                            <>
                              <span className="inline-flex items-center rounded bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-800 mb-3">
                                (Match Score: {score}){" "}
                              </span>
                            </>
                          ) : (
                            score
                          )}
                          <Highlighter
                            highlightClassName="YourHighlightClass"
                            searchWords={query.split(" ")}
                            autoEscape={true}
                            textToHighlight={truncatedText}
                          />
                        </p>
                      </div>
                    </div>
                  </li>
                );
              })}
        </ul>
      </main>
    </div>
  );
}
