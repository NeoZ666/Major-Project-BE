import express from 'express';
import bodyParser from 'body-parser';
import questionsRouter from './routes/google-document';
import userRouter from './routes/user';
import userResponseRouter from './routes/user-response';
import { corsConfig, REQUEST_FAILURE_MESSAGES, REQUEST_SUCCESS_MESSAGE, SECRET_KEY, SOCKET_EVENTS } from './common/constants';
import cors from "cors";
import { MongoClient, ServerApiVersion } from 'mongodb';
import { logger } from './common/pino';
import jwt from "jsonwebtoken";
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

// Initialize MongoDB client
const mongoClient = new MongoClient(process.env.MONGODB_URI!, {
  serverApi: {
    version: ServerApiVersion.v1,
    strict: false,
    deprecationErrors: true,
  }
});

const AUTHORISATION = "Authorization";
const SOCKET_CONNECTED = "Socket connected: ";
const app = express();

// Middleware setup
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization, Content-Length, X-Requested-With');
  
  // Handle preflight requests
  if (req.method === 'OPTIONS') {
    res.sendStatus(200);
  } else {
    next();
  }
});

app.use(cors(corsConfig));

// Authentication middleware
app.use((req: express.Request & { isUserAuth?: boolean; userId?: string }, res: express.Response, next: express.NextFunction) => {
  const authHeader = req.get(AUTHORISATION);
  if (!authHeader) {
    req.isUserAuth = false;
    return next();
  }

  const token = authHeader;
  let decodedToken: any;
  try {
    decodedToken = jwt.verify(token, SECRET_KEY);
  } catch (err) {
    req.isUserAuth = false;
    return next();
  }
  if (!decodedToken) {
    req.isUserAuth = false;
    return next();
  }
  req.userId = decodedToken.userId;
  req.isUserAuth = true;
  next();
});

// Routes
app.use(userRouter);
app.use(questionsRouter);
app.use(userResponseRouter);

async function startServer() {
  try {
    // Connect to MongoDB
    await mongoClient.connect();
    await mongoClient.db("admin").command({ ping: 1 });
    logger.info(REQUEST_SUCCESS_MESSAGE.DATABASE_CONNECTED_SUCCESSFULLY);

    // Start Express server
    const server = app.listen(process.env.PORT || 9000, () => {
      logger.info(REQUEST_SUCCESS_MESSAGE.APP_STARTED);
    });

    // Initialize Socket.IO
    const io = require('./common/Socket').init(server);
    io.on(SOCKET_EVENTS.CONNECTION, (socket: any) => {
      logger.info(SOCKET_CONNECTED, socket.id);
    });

    // Graceful shutdown handling
    const shutdown = async () => {
      logger.info('Shutdown signal received. Closing connections...');
      try {
        await mongoClient.close();
        server.close(() => {
          logger.info('Server closed successfully');
          process.exit(0);
        });
      } catch (error) {
        logger.error('Error during shutdown:', error);
        process.exit(1);
      }
    };

    process.on('SIGTERM', shutdown);
    process.on('SIGINT', shutdown);

  } catch (err) {
    logger.error(REQUEST_FAILURE_MESSAGES.ERROR_IN_CONNECTING_DB, err);
    logger.error(REQUEST_FAILURE_MESSAGES.APP_CRASHED);
    process.exit(1);
  }
}

// Start server with error handling
startServer().catch((err) => {
  logger.error('Failed to start server:', err);
  process.exit(1);
});

export default app;