const Mailgen = require('mailgen');
import type { NextApiRequest, NextApiResponse } from 'next';
const nodemailer = require('nodemailer');

export type EmailData = {
  email: string;
  htmlText: string;
  title?: string;
  host?: string;
  port?: string;
  userName?: string;
  userEmail?: string;
  password?: string;
};

export default function handler(
  req: NextApiRequest,
  res: NextApiResponse<EmailData>
) {
  const { email, htmlText, title, host, port, userName, userEmail, password } =
    req.body;

  const transporter = nodemailer.createTransport({
    host,
    port,
    secure: false,
    auth: {
      user: userEmail,
      pass: password,
    },
  });

  // Set up email data
  const mailOptions = {
    from: `"${userName}" <${userEmail}>`,
    to: email,
    subject: title ?? 'Invitation from Pracmanager',
    html: htmlText,
  };

  transporter
    .sendMail(mailOptions)
    .then(() => {
      return res.status(200).json(req.body);
    })
    .catch((error: any) => {
      return res.status(500).json(error);
    });
}
