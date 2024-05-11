/** @format */

import cors = require('cors');
import * as express from 'express';
import * as functions from 'firebase-functions';
const { fileParser } = require('express-multipart-file-parser');

const app = express();
app.use(cors());
app.use(
	fileParser({
		rawBodyOptions: {
			limit: '100mb',
		},
		busboyOptions: {
			limits: {
				fields: 2,
			},
		},
	})
);

// Set timeout to 10 minutes
const tenMinutesInMilliseconds = 10 * 60 * 1000;
app.use((req, res, next) => {
	req.setTimeout(tenMinutesInMilliseconds, () => {
		console.log('Request has timed out.');
		res.status(408).send('Request has timed out.');
	});
	next();
});

app.get('/hello', (req: any, res: any) => {
	res.send('Hello from Firebase!');
});

app.post('/webhook', async (req: any, res: any) => {
	try {
		const proxyResponse = await fetch(
			'http://13.53.134.33:8080/webhook/paypal',
			{
				method: 'POST',
				body: JSON.stringify(req.body),
			}
		);

		const responseData = await proxyResponse.json();
		res.status(200).json(responseData);
	} catch (err) {
		console.error(err);
		res.status(500).send(`Error ${String(err)}`);
	}
});

app.post('/upload', async (req: any, res: any) => {
	try {
		const { fieldname, originalname, encoding, mimetype, buffer } =
			req.files[0];
		console.log(fieldname, originalname, encoding, mimetype, buffer);

		const formData = new FormData();
		const videoBlob = new Blob([buffer], {
			type: 'video/mp4',
		}); // Adjust the MIME type as needed

		formData.append('video', videoBlob, `${Date.now()}${originalname}`);

		const proxyResponse = await fetch(
			'http://ec2-100-25-110-156.compute-1.amazonaws.com/predict',
			{
				method: 'POST',
				body: formData,
			}
		);

		const responseData = await proxyResponse.json();
		res.status(200).json(responseData);
	} catch (err) {
		console.error(err);
		res.status(500).send(`Error ${String(err)}`);
	}
});

const api = functions.https.onRequest(app);

export { api };
