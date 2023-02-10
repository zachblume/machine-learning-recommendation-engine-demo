// Next.js API route support: https://nextjs.org/docs/api-routes/introduction

export default async function handler(req, res) {
  let results = await fetch("http://34.86.228.54/?query=" + req.query.query);
  res.status(200).send(await results.text());
}
