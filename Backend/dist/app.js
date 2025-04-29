"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const body_parser_1 = __importDefault(require("body-parser"));
const google_document_1 = __importDefault(require("./routes/google-document"));
const user_1 = __importDefault(require("./routes/user"));
const user_response_1 = __importDefault(require("./routes/user-response"));
const constants_1 = require("./common/constants");
const cors_1 = __importDefault(require("cors"));
const mongodb_1 = require("mongodb");
const pino_1 = require("./common/pino");
const jsonwebtoken_1 = __importDefault(require("jsonwebtoken"));
const dotenv_1 = __importDefault(require("dotenv"));
// Load environment variables
dotenv_1.default.config();
// Initialize MongoDB client
const mongoClient = new mongodb_1.MongoClient(process.env.MONGODB_URI, {
    serverApi: {
        version: mongodb_1.ServerApiVersion.v1,
        strict: false,
        deprecationErrors: true,
    }
});
const AUTHORISATION = "Authorization";
const SOCKET_CONNECTED = "Socket connected: ";
const app = (0, express_1.default)();
// Middleware setup
app.use(body_parser_1.default.json());
app.use(body_parser_1.default.urlencoded({ extended: true }));
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS');
    res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization, Content-Length, X-Requested-With');
    // Handle preflight requests
    if (req.method === 'OPTIONS') {
        res.sendStatus(200);
    }
    else {
        next();
    }
});
app.use((0, cors_1.default)(constants_1.corsConfig));
// Authentication middleware
app.use((req, res, next) => {
    const authHeader = req.get(AUTHORISATION);
    if (!authHeader) {
        req.isUserAuth = false;
        return next();
    }
    const token = authHeader;
    let decodedToken;
    try {
        decodedToken = jsonwebtoken_1.default.verify(token, constants_1.SECRET_KEY);
    }
    catch (err) {
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
app.use(user_1.default);
app.use(google_document_1.default);
app.use(user_response_1.default);
function startServer() {
    return __awaiter(this, void 0, void 0, function* () {
        try {
            // Connect to MongoDB
            yield mongoClient.connect();
            yield mongoClient.db("admin").command({ ping: 1 });
            pino_1.logger.info("Moongoose connected successfully..." /* REQUEST_SUCCESS_MESSAGE.DATABASE_CONNECTED_SUCCESSFULLY */);
            // Start Express server
            const server = app.listen(process.env.PORT || 9000, () => {
                pino_1.logger.info("Express server is up and running" /* REQUEST_SUCCESS_MESSAGE.APP_STARTED */);
            });
            // Initialize Socket.IO
            const io = require('./common/Socket').init(server);
            io.on("connection" /* SOCKET_EVENTS.CONNECTION */, (socket) => {
                pino_1.logger.info(SOCKET_CONNECTED, socket.id);
            });
            // Graceful shutdown handling
            const shutdown = () => __awaiter(this, void 0, void 0, function* () {
                pino_1.logger.info('Shutdown signal received. Closing connections...');
                try {
                    yield mongoClient.close();
                    server.close(() => {
                        pino_1.logger.info('Server closed successfully');
                        process.exit(0);
                    });
                }
                catch (error) {
                    pino_1.logger.error('Error during shutdown:', error);
                    process.exit(1);
                }
            });
            process.on('SIGTERM', shutdown);
            process.on('SIGINT', shutdown);
        }
        catch (err) {
            pino_1.logger.error("Unable to connect the monog-db database" /* REQUEST_FAILURE_MESSAGES.ERROR_IN_CONNECTING_DB */, err);
            pino_1.logger.error("App crashed" /* REQUEST_FAILURE_MESSAGES.APP_CRASHED */);
            process.exit(1);
        }
    });
}
// Start server with error handling
startServer().catch((err) => {
    pino_1.logger.error('Failed to start server:', err);
    process.exit(1);
});
exports.default = app;
