"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.SECRET_KEY = exports.UNAUTHORIZED_ACCESS = exports.corsConfig = void 0;
exports.corsConfig = {
    origin: true, // Allow all origins temporarily for debugging
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: [
        "Authorization",
        "X-Requested-With",
        "Content-Type",
        "x-auth-token",
        "Access-Control-Allow-Origin",
        "Access-Control-Allow-Headers"
    ],
    exposedHeaders: ["Authorization"],
    maxAge: 86400,
    credentials: true,
    optionsSuccessStatus: 200,
    preflightContinue: false
};
exports.UNAUTHORIZED_ACCESS = "Unauthorised resource access..!";
exports.SECRET_KEY = "somesupersecretsecret";
