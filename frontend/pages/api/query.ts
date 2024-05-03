import { Environment } from '@/modules/environment';
import type { NextApiRequest, NextApiResponse } from 'next';

const NodeCache = require('node-cache');

/**
 * Api for adding new items to the list
 * @param req
 * @param res
 * @returns
 */
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<any>
) {
  try {
    if (
      ['http://localhost:3000', 'https://smart-sheet-app.netlify.app'].includes(
        req?.headers?.referer ?? ''
      )
    ) {
      res.status(500).send('Something went wrong');
      return;
    }

    const body = JSON.stringify(req.body);
    const request = await fetch(`${Environment.hasuraUrl}/`, {
      body,
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
        'x-hasura-admin-secret': Environment.hasuraSecret,
      },
    });

    const response = await request.json();
    res.status(200).json({ ...response });
    return;
  } catch (err) {
    res.status(500).send('Error');
  }
}
