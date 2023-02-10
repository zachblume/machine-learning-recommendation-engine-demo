import { NextResponse } from "next/server";

export const config = {
  runtime: "edge", // this is a pre-requisite
  regions: ["iad1"], // only execute this function on iad1
};

export default (req) => {
  let results = await fetch("http://34.86.228.54/?query=" + req.query.query);
  return NextResponse.json(results);
};
