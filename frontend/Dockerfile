FROM node:22-alpine

WORKDIR /app
RUN chown node:node /app

# Upgrade OS packages to fix potential vulnerabilities
RUN apk update && apk upgrade --no-cache

COPY --chown=node:node package.json /app/

# Update npm to latest globally as root to fix internal dependencies like glob
RUN npm install -g npm@latest

# Install dependencies as the node user to ensure permissions
USER node
RUN npm install

COPY --chown=node:node . /app

EXPOSE 3000

CMD ["node", "server.js"]
