package data.scripts.utils.network_training;

import data.scripts.utils.ARR_ShipData;
import data.scripts.utils.ARR_ShipDataManager;
import data.scripts.utils.network.SimpleSocketClient;

public class ARR_StateSender {

    private static final String SERVER_IP = "127.0.0.1";
    private static final int SERVER_PORT = 65432;

    private final SimpleSocketClient client;

    public ARR_StateSender() {
        client = new SimpleSocketClient(SERVER_IP, SERVER_PORT);
    }

    public boolean connect() {
        return client.connect();
    }

    public boolean isConnected() {
        return client.isConnected();
    }

    public void disconnect() {
        client.disconnect();
    }

    public void sendAllStates() {
        if (!client.isConnected()) return;

        for (ARR_ShipData ship : ARR_ShipDataManager.getInstance().getAll()) {
            String line = ship.shipId + ","
                    + ship.getLocation().x + ","
                    + ship.getLocation().y + ","
                    + ship.getVelocity().x + ","
                    + ship.getVelocity().y + ","
                    + ship.getAngle() + ","
                    + ship.getAngularVelocity();
            client.send(line);
        }
        client.send("END");
    }

    public boolean hasMessage() {
        return client.hasMessage();
    }

    public String getMessage() {
        return client.getMessage();
    }
}
