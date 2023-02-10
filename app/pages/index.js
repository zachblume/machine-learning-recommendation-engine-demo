import Head from "next/head";
import Image from "next/image";
import styles from "../styles/Home.module.css";

export default function Home() {
  return (
    <div className={styles.container}>
      <Head>
        <title>Clinical trial theMedNet demo for interview</title>
        <meta name="description" content="Generated by create next app" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main>
        <h1>
          A series of questions and their most closely matching clinical trials
          using medBERT
        </h1>
        {examples.map((example) => (
          <>
            <div className="card">
              <h2>{example.title}</h2>
              <p>{example.text}</p>
              <div className="best-trial-match-fulltext">
                {example.bestTrialMatch}
              </div>
            </div>
          </>
        ))}
        <h1>Search the clinical trials using medBERT</h1>
        <input type="text" id="searchFor" className="" />
        <ul className="search-results"></ul>
      </main>
    </div>
  );
}
