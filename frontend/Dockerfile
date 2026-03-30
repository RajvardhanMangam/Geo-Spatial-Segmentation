FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json* ./

# --legacy-peer-deps is required due to peer dep conflicts (leaflet, proj4, react versions)
RUN npm install --legacy-peer-deps

COPY . .

ENV REACT_APP_API_URL=http://localhost:8000
ENV REACT_APP_WS_URL=ws://localhost:8000

RUN npm run build

# Serve the built app
RUN npm install -g serve
CMD ["serve", "-s", "build", "-l", "3000"]
