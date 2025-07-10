/**
 * Luffy Bolta Hai - Main Entry Point
 * 
 * This is the main entry point for the Luffy Bolta Hai application.
 * The application is built using a modular architecture with separate
 * modules for different concerns (API, UI, services, utils).
 */

console.log('app.js loaded');

// Import the main application class
import { LuffyBoltHaiApp } from './modules/app.js';

// Initialize the application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM fully loaded, initializing app...');
    try {
        window.app = new LuffyBoltHaiApp();
        console.log('App initialized successfully');
    } catch (error) {
        console.error('Error initializing app:', error);
    }
});
