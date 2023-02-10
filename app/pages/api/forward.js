// Next.js API route support: https://nextjs.org/docs/api-routes/introduction

export default async function handler(req, res) {
  let results = await fetch(req.query.url);
  res.status(200).send(await results.text());
}
