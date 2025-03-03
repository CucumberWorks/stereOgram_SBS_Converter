# How to Get a Discord Bot Token

This guide will walk you through the process of creating a Discord application, setting up a bot user, and obtaining your bot token.

## Step 1: Access the Discord Developer Portal

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
2. Log in with your Discord account if you're not already logged in

## Step 2: Create a New Application

1. Click the "New Application" button in the top-right corner
2. Enter a name for your application (e.g., "Stereo3D Bot")
3. Accept the terms of service
4. Click "Create"

## Step 3: Create a Bot User

1. In the left sidebar, click on "Bot"
2. Click the "Add Bot" button
3. Confirm by clicking "Yes, do it!"

## Step 4: Configure Bot Settings

1. Under the "TOKEN" section, click "Reset Token" to generate a new token (You'll need to confirm this action)
2. Copy your token and store it securely (you won't be able to see it again!)
3. Under "Privileged Gateway Intents", enable:
   - "Presence Intent"
   - "Server Members Intent"
   - "Message Content Intent" (very important for the bot to see image attachments!)

## Step 5: Set Up Bot Permissions

1. In the left sidebar, click on "OAuth2" and then "URL Generator"
2. Under "Scopes", select "bot"
3. Under "Bot Permissions", select:
   - "Send Messages"
   - "Attach Files"
   - "Read Message History"
   - "Add Reactions"
   - "Use Slash Commands" (if you want to add slash commands in the future)

## Step 6: Invite Your Bot to Your Server

1. Copy the generated URL at the bottom of the page
2. Paste it in your web browser
3. Select the server you want to add the bot to
4. Click "Authorize"
5. Complete the captcha if prompted

## Step 7: Add Bot Token to Your Project

1. Create a `.env` file in the root directory of the project
2. Add your token:
   ```
   DISCORD_BOT_TOKEN=your_token_here
   ```
3. Save the file
4. Make sure not to share this file or commit it to public repositories

## Important Security Notes

- **Never share your bot token!** Anyone with your token can control your bot
- Don't include your token directly in your code
- If you accidentally expose your token, go back to the Discord Developer Portal and reset it immediately
- Add your `.env` file to `.gitignore` to prevent accidentally committing it
- If you're publishing your bot code, make sure to use environment variables or a configuration file that's not included in your repository

## Troubleshooting

- If your bot doesn't come online, check if you've properly copied the token
- If your bot can't see messages, make sure you've enabled "Message Content Intent"
- If your bot can't join servers, verify you've set up the correct permissions 